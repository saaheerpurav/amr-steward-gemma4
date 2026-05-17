"""
app.py — AMR-Steward FastAPI server.

Uses a module-level AMREnvironment singleton for /reset and /step.
openenv-core's create_app creates a fresh instance per request (breaking
session state), so we own the reset/step/state/health routes directly.

Run locally:
  uvicorn app:app --reload --port 7860

HuggingFace Spaces:
  CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env import AMRAction, AMREnvironment
from env.models import PatientCase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("amr_steward.app")

# ---------------------------------------------------------------------------
# Module-level singleton — persists across requests on the same worker
# ---------------------------------------------------------------------------
_ENV = AMREnvironment()

# Warm up data loaders at import time so the first request isn't cold.
try:
    from env.environment import _get_drug_properties, _get_eucast, _get_idsa
    _get_idsa()
    _get_drug_properties()
    _get_eucast()
    logger.info("Data layers loaded successfully.")
except Exception as exc:
    logger.warning("Data pre-load warning: %s", exc)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AMR-Steward",
    description="RL environment for antimicrobial stewardship — trains Gemma 4 to prescribe correctly for drug-resistant infections.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE = Path(__file__).resolve().parent

# Serve demo_web/ assets (JS, CSS, images if any) under /demo-static/
_DEMO_DIR = _BASE / "demo_web"
if _DEMO_DIR.exists():
    app.mount("/demo-static", StaticFiles(directory=str(_DEMO_DIR)), name="demo_static")

_STATIC_DIR = _BASE / "static"

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    curriculum_level: int = 1
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    patient: Optional[Dict[str, Any]] = None  # inject a specific PatientCase for reproducible eval


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Core environment endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", tags=["Environment Control"])
def reset(body: ResetRequest = ResetRequest()) -> JSONResponse:
    """Start a new episode. Returns the initial observation."""
    try:
        injected_patient: PatientCase | None = None
        if body.patient is not None:
            try:
                injected_patient = PatientCase(**body.patient)
            except Exception as exc:
                logger.warning("Invalid patient payload in /reset — using random patient: %s", exc)

        obs = _ENV.reset(
            seed=body.seed,
            episode_id=body.episode_id,
            curriculum_level=body.curriculum_level,
            patient=injected_patient,
        )
        return JSONResponse({
            "observation": obs.model_dump(),
            "reward": None,
            "done": False,
        })
    except Exception as exc:
        logger.exception("reset() failed")
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.post("/step", tags=["Environment Control"])
def step(body: StepRequest) -> JSONResponse:
    """Execute an action and return the resulting observation + reward."""
    try:
        action = AMRAction(**body.action)
        obs = _ENV.step(action)
        return JSONResponse({
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        })
    except Exception as exc:
        logger.exception("step() failed")
        return JSONResponse({"detail": str(exc)}, status_code=500)


@app.get("/state", tags=["Environment Control"])
def state() -> JSONResponse:
    """Return current episode state."""
    return JSONResponse(_ENV.state.model_dump())


@app.get("/health", tags=["Environment Control"])
def health() -> JSONResponse:
    """Liveness check."""
    return JSONResponse({"status": "healthy"})


@app.get("/api/jepa-rankings", tags=["JEPA World Model"])
def jepa_rankings() -> JSONResponse:
    """Read-only: return JEPA predicted information-gain scores for each available tool
    given the current episode state.

    This endpoint exposes the world model's latent-space predictions so the demo
    UI can visualise them as 'World Model Confidence' bars next to each clue card.

    Safe to call at any time — it reads state but never mutates it.
    Returns an empty list if no episode is active.
    """
    try:
        from env.world_model import AVAILABLE_TOOLS, AMRWorldModel
        wm = _get_world_model()  # already-loaded singleton
        patient = _ENV.current_patient
        if patient is None:
            return JSONResponse({"rankings": [], "note": "No active episode — call /reset first."})

        known_state = wm.encode_known_state(
            list(_ENV._state.tool_results),
            patient.__dict__,
        )
        rankings = wm.get_test_rankings(known_state, AVAILABLE_TOOLS)
        return JSONResponse({
            "rankings": [{"tool": tool, "score": round(score, 4)} for tool, score in rankings],
            "episode_id": _ENV._state.episode_id,
            "tools_called": len(_ENV._state.tool_results),
        })
    except Exception as exc:
        logger.exception("jepa_rankings() failed")
        return JSONResponse({"detail": str(exc), "rankings": []}, status_code=500)


# ---------------------------------------------------------------------------
# WhatsApp webhook (Twilio)
# ---------------------------------------------------------------------------

@app.post("/whatsapp", include_in_schema=False)
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(default=""),
) -> Response:
    from whatsapp_bot import handle_whatsapp
    xml = handle_whatsapp(from_number=From, body=Body)
    return Response(content=xml, media_type="application/xml")


# ---------------------------------------------------------------------------
# Demo + Landing page
# ---------------------------------------------------------------------------

@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo() -> HTMLResponse:
    """Interactive browser demo — cinematic act-by-act experience."""
    demo_file = _DEMO_DIR / "index.html"
    if demo_file.exists():
        return HTMLResponse(content=demo_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<p>Demo not found.</p>", status_code=404)


# ---------------------------------------------------------------------------
# Landing page — served last so API routes always take priority
# html=True serves index.html for / and falls back to it for unmatched paths
# ---------------------------------------------------------------------------
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="landing")
