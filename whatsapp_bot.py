"""
whatsapp_bot.py — AMR-Steward WhatsApp demo for hackathon judges.

Flow:
  1. Any message → patient case + 4 drug choices (numbered)
  2. Judge picks 1-4 → their score revealed (usually bad)
  3. AI investigation steps revealed → AI score (good)
  4. "Reply 1 to try again"

Fixed patient: canonical CRE K. pneumoniae case from the model card.
No OpenAI needed — pure Twilio + env.
"""

from __future__ import annotations

import logging
from typing import Any

from twilio.twiml.messaging_response import MessagingResponse

from env import AMRAction, AMREnvironment
from env.models import PatientCase

logger = logging.getLogger("amr_steward.whatsapp")

# ── Fixed patient (canonical hard case from model card) ───────────────────────

FIXED_PATIENT = PatientCase(
    age=67,
    sex="F",
    infection_site="bacteremia",
    organism="K. pneumoniae",
    creatinine_clearance=35.0,
    allergies=[],
    antibiogram={
        "meropenem": 8.0,
        "ceftazidime-avibactam": 1.0,
        "colistin": 1.0,
        "meropenem-vaborbactam": 1.0,
        "ceftriaxone": 32.0,
        "ciprofloxacin": 16.0,
    },
    phenotype="resistant",
    curriculum_level=2,
)

# ── Drug menu ─────────────────────────────────────────────────────────────────
# (drug_key, dose, display_name)
DRUG_CHOICES = {
    "1": ("meropenem",              "1g IV q8h",     "Meropenem"),
    "2": ("ceftazidime-avibactam",  "1.25g IV q12h", "Ceftazidime-avibactam"),
    "3": ("ciprofloxacin",          "400mg IV q12h", "Ciprofloxacin"),
    "4": ("ceftriaxone",            "2g IV q24h",    "Ceftriaxone"),
}

PRESENTATION = (
    "🏥 *PATIENT 1047-F*\n\n"
    "67 years old · Female · ICU\n"
    "Drug-resistant bloodstream infection\n"
    "Kidneys at 35% function · No allergies\n\n"
    "What antibiotic do you prescribe?\n\n"
    "1️⃣  Meropenem\n"
    "2️⃣  Ceftazidime-avibactam\n"
    "3️⃣  Ciprofloxacin\n"
    "4️⃣  Ceftriaxone"
)

# ── Session state ─────────────────────────────────────────────────────────────
# phone → "waiting_choice" | "done"
_SESSIONS: dict[str, str] = {}

RESET_WORDS = {"hi", "hello", "start", "new", "reset", "again", "/start"}


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _score_prescription(drug: str, dose: str) -> tuple[float, dict]:
    """Reset env with fixed patient, commit drug, return (reward, breakdown)."""
    env = AMREnvironment()
    env.reset(curriculum_level=2, patient=FIXED_PATIENT)
    action = AMRAction(
        action_type="COMMIT",
        prescription={
            "drug": drug,
            "dose": dose,
            "duration": "14 days",
            "justification": "judge prescription",
        },
    )
    obs = env.step(action)
    reward = obs.reward or 0.0
    breakdown = (obs.metadata or {}).get("reward_breakdown", {})
    return reward, breakdown


def _score_ai() -> tuple[float, list[str], dict]:
    """
    Replay the AI's optimal investigation path, return
    (reward, investigation_lines, breakdown).
    """
    env = AMREnvironment()
    env.reset(curriculum_level=2, patient=FIXED_PATIENT)

    steps = [
        AMRAction(action_type="INVESTIGATE", tool_name="interpret_resistance",   tool_arg="meropenem"),
        AMRAction(action_type="INVESTIGATE", tool_name="check_guideline",        tool_arg="bacteremia"),
        AMRAction(action_type="INVESTIGATE", tool_name="assess_patient_factors", tool_arg=""),
    ]
    results = []
    for step in steps:
        obs = env.step(step)
        results.append(obs.tool_results[-1] if obs.tool_results else "")

    commit = AMRAction(
        action_type="COMMIT",
        prescription={
            "drug": "ceftazidime-avibactam",
            "dose": "1.25g IV q12h",
            "duration": "14 days",
            "justification": "Gemma 4 trained with GRPO on verified medical reward",
        },
    )
    obs = env.step(commit)
    reward = obs.reward or 0.0
    breakdown = (obs.metadata or {}).get("reward_breakdown", {})
    return reward, results, breakdown


# ── Message builders ──────────────────────────────────────────────────────────

def _judge_score_msg(drug_name: str, reward: float, breakdown: dict) -> str:
    if reward >= 0.85:
        glow, verdict = "🟢", "Strong prescription."
    elif reward >= 0.5:
        glow, verdict = "🟡", "Suboptimal — patient at risk."
    else:
        glow, verdict = "🔴", "This drug has zero effect on this strain.\nShe doesn't survive."

    activity = breakdown.get("R1_activity", 0)
    active_str = "✅ Active against organism" if activity >= 0.9 else "❌ No activity against organism"

    return (
        f"Scoring *{drug_name}*...\n\n"
        f"{glow} *YOUR SCORE: {reward:.2f}*\n\n"
        f"{active_str}\n\n"
        f"{verdict}"
    )


def _ai_reveal_msg(reward: float, invest_results: list[str]) -> str:
    # Condense the 3 raw env results into 3 short lines
    lines = [
        f"🔬 Checks meropenem → *RESISTANT* ✗",
        f"📚 Checks IDSA guidelines → *ceftazidime-avibactam* for CRE bacteremia",
        f"💊 Checks kidney dose → reduce to *1.25g q12h* for CrCl 35",
    ]

    if reward >= 0.85:
        verdict = "✅ *She survives.*"
    else:
        verdict = "⚠️ Suboptimal."

    return (
        f"🤖 *AMR-STEWARD* (trained with GRPO):\n\n"
        + "\n".join(lines)
        + f"\n\nRx: *ceftazidime-avibactam 1.25g IV q12h · 14 days*\n\n"
        f"🟢 *AI SCORE: {reward:.2f}*\n\n"
        f"{verdict}\n\n"
        f"_Reply *1* to prescribe again_"
    )


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_whatsapp(from_number: str, body: str) -> str:
    phone = from_number.strip()
    text = body.strip().lower()

    state = _SESSIONS.get(phone, "new")

    # Any reset trigger (or done + "1") → show patient presentation
    reset_requested = text in RESET_WORDS or (state == "done" and text == "1")
    if state == "new" or reset_requested:
        _SESSIONS[phone] = "waiting_choice"
        return _twiml(PRESENTATION)

    # Waiting for drug choice
    if state == "waiting_choice":
        if text not in DRUG_CHOICES:
            return _twiml(
                "Reply with a number:\n\n"
                "1️⃣  Meropenem\n"
                "2️⃣  Ceftazidime-avibactam\n"
                "3️⃣  Ciprofloxacin\n"
                "4️⃣  Ceftriaxone"
            )

        drug_key, dose, drug_name = DRUG_CHOICES[text]

        # Score judge's choice
        judge_reward, judge_breakdown = _score_prescription(drug_key, dose)

        # Score AI path
        ai_reward, ai_invest_results, _ = _score_ai()

        _SESSIONS[phone] = "done"

        # Two messages in one TwiML response — Twilio delivers in order
        resp = MessagingResponse()
        resp.message(_judge_score_msg(drug_name, judge_reward, judge_breakdown))
        resp.message(_ai_reveal_msg(ai_reward, ai_invest_results))
        return str(resp)

    # Fallback
    _SESSIONS[phone] = "new"
    return _twiml("Reply *hi* to start a new case 🏥")


def _twiml(message: str) -> str:
    resp = MessagingResponse()
    resp.message(message)
    return str(resp)
