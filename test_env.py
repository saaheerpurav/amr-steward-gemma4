"""
test_env.py — Integration smoke test for AMR-Steward (OpenEnv compliant).
Run: python test_env.py
"""
import sys
sys.path.insert(0, ".")

from env import AMRAction, AMREnvironment, AMRState, PatientCase

PASS = "[PASS]"
FAIL = "[FAIL]"


def make_demo_env() -> AMREnvironment:
    """Env pre-loaded with the canonical 67F CRE bacteremia demo patient."""
    env = AMREnvironment()
    env.current_patient = PatientCase(
        age=67, sex="F", infection_site="bacteremia",
        organism="K. pneumoniae", creatinine_clearance=35.0,
        allergies=[],
        antibiogram={"meropenem": 8.0, "ceftazidime-avibactam": 1.0, "colistin": 1.0},
        phenotype="resistant", curriculum_level=1,
    )
    env._state = AMRState(
        episode_id="demo-fixture",
        step_count=0,
        curriculum_level=1,
        budget_remaining=5,
        done=False,
        patient=env.current_patient.__dict__.copy(),
        tool_results=[],
        called_tools=[],
        dense_accum=0.0,
        tool_history=[],
    )
    return env


def test_reset():
    env = AMREnvironment()
    obs = env.reset(curriculum_level=1)
    assert obs.budget_remaining == 5, "budget should be 5 for level 1"
    assert not obs.done
    assert env.state.episode_id, "state.episode_id must be set"
    print(PASS + " reset() level=1 | budget=5 | episode_id=" + env.state.episode_id[:8])

    obs2 = env.reset(curriculum_level=3, episode_id="custom-id")
    assert obs2.budget_remaining == 3
    assert env.state.episode_id == "custom-id"
    print(PASS + " reset() level=3 with custom episode_id | budget=3")


def test_tools():
    env = make_demo_env()

    obs = env.step(AMRAction(action_type="INVESTIGATE",
                             tool_name="interpret_resistance", tool_arg="meropenem"))
    assert not obs.done
    assert "Resistant" in obs.tool_results[-1] or "MIC" in obs.tool_results[-1]
    assert obs.budget_remaining == 4
    print(PASS + " interpret_resistance meropenem | "
          + obs.tool_results[-1][:80].replace("\n", " "))

    obs = env.step(AMRAction(action_type="INVESTIGATE",
                             tool_name="check_guideline", tool_arg="bacteremia"))
    assert not obs.done
    assert ("ceftazidime-avibactam" in obs.tool_results[-1].lower()
            or "IDSA" in obs.tool_results[-1])
    print(PASS + " check_guideline bacteremia | "
          + obs.tool_results[-1][:80].replace("\n", " "))

    obs = env.step(AMRAction(action_type="INVESTIGATE",
                             tool_name="assess_patient_factors"))
    assert not obs.done
    assert "CrCl" in obs.tool_results[-1] or "Renal" in obs.tool_results[-1]
    print(PASS + " assess_patient_factors | "
          + obs.tool_results[-1][:80].replace("\n", " "))


def test_correct_prescription():
    env = make_demo_env()
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="interpret_resistance", tool_arg="meropenem"))
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="check_guideline", tool_arg="bacteremia"))
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="assess_patient_factors"))

    obs = env.step(AMRAction(action_type="COMMIT", prescription={
        "drug": "ceftazidime-avibactam",
        "dose": "1.25g IV q8h",
        "duration": "14 days",
        "justification": "CRE K. pneumoniae, IDSA first-line, renal-adjusted",
    }))
    assert obs.done
    bd = env.state.last_reward_breakdown
    assert bd["R1_activity"] == 1.0, "R1 should be 1.0"
    assert bd["R2_guideline"] >= 0.5, "R2 should be >= 0.5"
    assert bd["R5_efficiency"] > 0.0, "R5 should be > 0 — tools were called"
    assert obs.reward >= 0.5, "Total reward should be >= 0.5"
    print(PASS + f" CORRECT Rx reward={obs.reward}"
          f" | R1={bd['R1_activity']} R2={bd['R2_guideline']}"
          f" R3={bd['R3_stewardship']} R4={bd['R4_dose']} R5={bd['R5_efficiency']}")


def test_wrong_prescription():
    env = make_demo_env()
    obs = env.step(AMRAction(action_type="COMMIT", prescription={
        "drug": "meropenem",
        "dose": "1g IV q8h",
        "duration": "14 days",
        "justification": "blind guess",
    }))
    assert obs.done
    bd = env.state.last_reward_breakdown
    assert bd["R1_activity"] == 0.0, "R1 must be 0 for resistant drug"
    assert bd["R3_stewardship"] == 0.0, "R3 must be 0 when R1 is 0"
    assert bd["R5_efficiency"] == 0.0, "R5 must be 0 — no tools called"
    assert obs.reward < 0.4, "Wrong Rx should have low reward"
    print(PASS + f" WRONG Rx (meropenem/CRE) reward={obs.reward}"
          f" | R1={bd['R1_activity']} (expect 0.0)")


def test_budget_exhaustion():
    env = make_demo_env()
    env._state.budget_remaining = 1
    obs = env.step(AMRAction(action_type="INVESTIGATE",
                             tool_name="assess_patient_factors"))
    assert obs.done, "Budget exhausted -> done=True"
    assert obs.reward < 0, "Budget exhaustion should give negative reward"
    print(PASS + f" budget exhaustion | reward={obs.reward} done={obs.done}")


def test_invalid_action():
    try:
        AMRAction(action_type="INVALID_TYPE")
        print(FAIL + " AMRAction validator should have rejected 'INVALID_TYPE'")
    except Exception:
        print(PASS + " invalid action_type rejected by Pydantic validator")


def test_state_property():
    env = make_demo_env()
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="interpret_resistance", tool_arg="meropenem"))
    s = env.state
    assert s.step_count == 1
    assert s.budget_remaining == 4
    assert len(s.tool_results) == 1
    assert s.episode_id == "demo-fixture"
    print(PASS + f" state property | step={s.step_count} budget={s.budget_remaining}"
          f" episode_id={s.episode_id}")


def test_app_import():
    from app import app
    routes = {r.path for r in app.routes}
    assert "/" in routes
    assert any("/reset" in r for r in routes), "OpenEnv should expose /reset"
    print(PASS + f" app.py imports | {len(routes)} routes registered (OpenEnv + /)")


if __name__ == "__main__":
    print("=" * 60)
    print("AMR-Steward Integration Tests (OpenEnv-compliant)")
    print("=" * 60)
    errors = []
    for fn in [test_reset, test_tools, test_correct_prescription,
               test_wrong_prescription, test_budget_exhaustion,
               test_invalid_action, test_state_property, test_app_import]:
        try:
            fn()
        except Exception as e:
            print(FAIL + " " + fn.__name__ + " => " + repr(e))
            errors.append(fn.__name__)
    print("=" * 60)
    if errors:
        print("FAILED: " + ", ".join(errors))
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
