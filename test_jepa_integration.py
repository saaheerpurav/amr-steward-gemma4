"""
test_jepa_integration.py — Rigorous tests for JEPA reward shaping,
latent consistency bonus, action-to-key mapping, and cap enforcement.
Run: python test_jepa_integration.py
"""
import sys
sys.path.insert(0, ".")

from env import AMRAction, AMREnvironment, AMRState, PatientCase

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []


def _log(name, passed, msg=""):
    tag = PASS if passed else FAIL
    print(f"{tag} {name}" + (f" | {msg}" if msg else ""))
    if not passed:
        errors.append(name)


def make_demo_env() -> AMREnvironment:
    env = AMREnvironment()
    env.current_patient = PatientCase(
        age=67, sex="F", infection_site="bacteremia",
        organism="K. pneumoniae", creatinine_clearance=35.0,
        allergies=[],
        antibiogram={"meropenem": 8.0, "ceftazidime-avibactam": 1.0, "colistin": 1.0},
        phenotype="resistant", curriculum_level=1,
    )
    env._state = AMRState(
        episode_id="jepa-test",
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


# ---------------------------------------------------------------------------
# T1: _action_to_jepa_key mapping
# ---------------------------------------------------------------------------
def test_action_to_jepa_key():
    from env.environment import _action_to_jepa_key

    assert _action_to_jepa_key("interpret_resistance", "meropenem") == "interpret_resistance_meropenem"
    assert _action_to_jepa_key("interpret_resistance", "MEROPENEM") == "interpret_resistance_meropenem"
    assert _action_to_jepa_key("check_guideline", "bacteremia") == "check_guideline_bacteremia"
    assert _action_to_jepa_key("assess_patient_factors", None) == "assess_patient_factors"
    assert _action_to_jepa_key(None, None) == ""
    assert _action_to_jepa_key("interpret_resistance", None) == "interpret_resistance"
    _log("action_to_jepa_key", True, "all 6 mappings correct")


# ---------------------------------------------------------------------------
# T2: JEPA info_gain and actual_delta appear in tool_history
# ---------------------------------------------------------------------------
def test_jepa_fields_in_tool_history():
    env = make_demo_env()
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="interpret_resistance", tool_arg="meropenem"))
    entry = env.state.tool_history[0]

    has_gain = "jepa_info_gain" in entry
    has_delta = "actual_delta" in entry
    has_bonus = "consistency_bonus" in entry
    gain_valid = isinstance(entry.get("jepa_info_gain"), float)
    delta_valid = isinstance(entry.get("actual_delta"), float)

    _log("jepa_fields_in_tool_history", has_gain and has_delta and has_bonus and gain_valid and delta_valid,
         f"jepa_info_gain={entry.get('jepa_info_gain')} actual_delta={entry.get('actual_delta')} "
         f"consistency_bonus={entry.get('consistency_bonus')}")


# ---------------------------------------------------------------------------
# T3: JEPA info_gain is bounded [0.0, 1.0]
# ---------------------------------------------------------------------------
def test_jepa_info_gain_bounded():
    env = make_demo_env()
    for tool, arg in [
        ("interpret_resistance", "meropenem"),
        ("check_guideline", "bacteremia"),
        ("assess_patient_factors", None),
    ]:
        env2 = make_demo_env()
        env2.step(AMRAction(action_type="INVESTIGATE", tool_name=tool, tool_arg=arg))
        gain = env2.state.tool_history[0]["jepa_info_gain"]
        ok = 0.0 <= gain <= 1.0
        _log(f"jepa_info_gain_bounded [{tool}]", ok, f"gain={gain}")


# ---------------------------------------------------------------------------
# T4: actual_delta is bounded [0.0, 1.0]
# ---------------------------------------------------------------------------
def test_actual_delta_bounded():
    env = make_demo_env()
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="check_guideline", tool_arg="bacteremia"))
    delta = env.state.tool_history[0]["actual_delta"]
    ok = 0.0 <= delta <= 1.0
    _log("actual_delta_bounded", ok, f"delta={delta}")


# ---------------------------------------------------------------------------
# T5: Dense accumulator never exceeds _DENSE_CAP
# ---------------------------------------------------------------------------
def test_dense_cap_enforced():
    env = make_demo_env()
    env._state.budget_remaining = 10  # give extra budget

    # Call 5 distinct tools to maximise accumulation
    actions = [
        AMRAction(action_type="INVESTIGATE", tool_name="interpret_resistance", tool_arg="meropenem"),
        AMRAction(action_type="INVESTIGATE", tool_name="interpret_resistance", tool_arg="ceftazidime-avibactam"),
        AMRAction(action_type="INVESTIGATE", tool_name="interpret_resistance", tool_arg="colistin"),
        AMRAction(action_type="INVESTIGATE", tool_name="check_guideline", tool_arg="bacteremia"),
        AMRAction(action_type="INVESTIGATE", tool_name="assess_patient_factors"),
    ]
    for a in actions:
        env.step(a)

    accum = env._dense_accum
    cap = env._DENSE_CAP
    ok = accum <= cap + 1e-9
    _log("dense_cap_enforced", ok, f"accum={accum:.4f} cap={cap}")


# ---------------------------------------------------------------------------
# T6: JEPA-scaled bonus > 0 for novel tool call (scale >= 0.5 means bonus > 0)
# ---------------------------------------------------------------------------
def test_jepa_scaled_bonus_positive():
    env = make_demo_env()
    obs = env.step(AMRAction(action_type="INVESTIGATE",
                             tool_name="interpret_resistance", tool_arg="meropenem"))
    # reward = JEPA-scaled dense + consistency; should be > 0
    ok = obs.reward is not None and obs.reward > 0.0
    _log("jepa_scaled_bonus_positive", ok, f"step_reward={obs.reward}")


# ---------------------------------------------------------------------------
# T7: Repeated same tool call gives 0 dense bonus (but may get tiny consistency)
# ---------------------------------------------------------------------------
def test_repeated_tool_no_dense_bonus():
    env = make_demo_env()
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="assess_patient_factors"))
    accum_after_first = env._dense_accum

    obs = env.step(AMRAction(action_type="INVESTIGATE",
                             tool_name="assess_patient_factors"))
    accum_after_second = env._dense_accum

    # Dense novel bonus should be 0; only consistency bonus (tiny) may be added
    dense_inc = accum_after_second - accum_after_first
    ok = dense_inc < env._DENSE_NOVEL_TOOL  # no full novel bonus, only tiny consistency
    _log("repeated_tool_no_dense_bonus", ok,
         f"dense_inc={dense_inc:.4f} (expect < {env._DENSE_NOVEL_TOOL})")


# ---------------------------------------------------------------------------
# T8: consistency_bonus is non-negative and bounded by _CONSISTENCY_SCALE
# ---------------------------------------------------------------------------
def test_consistency_bonus_bounded():
    env = make_demo_env()
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="check_guideline", tool_arg="bacteremia"))
    bonus = env.state.tool_history[0]["consistency_bonus"]
    ok = 0.0 <= bonus <= env._CONSISTENCY_SCALE + 1e-9
    _log("consistency_bonus_bounded", ok,
         f"bonus={bonus:.4f} scale={env._CONSISTENCY_SCALE}")


# ---------------------------------------------------------------------------
# T9: unknown tool_arg gives graceful 0 JEPA gain (no crash)
# ---------------------------------------------------------------------------
def test_unknown_tool_arg_graceful():
    env = make_demo_env()
    try:
        obs = env.step(AMRAction(action_type="INVESTIGATE",
                                 tool_name="interpret_resistance",
                                 tool_arg="nonexistent_drug_xyz"))
        gain = env.state.tool_history[0]["jepa_info_gain"]
        ok = gain == 0.0 and obs is not None
        _log("unknown_tool_arg_graceful", ok, f"gain={gain} (expect 0.0, no crash)")
    except Exception as e:
        _log("unknown_tool_arg_graceful", False, f"crashed: {e}")


# ---------------------------------------------------------------------------
# T10: Full episode flow — JEPA data accumulates correctly across steps
# ---------------------------------------------------------------------------
def test_full_episode_jepa_accumulation():
    env = make_demo_env()

    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="interpret_resistance", tool_arg="meropenem"))
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="check_guideline", tool_arg="bacteremia"))
    env.step(AMRAction(action_type="INVESTIGATE",
                       tool_name="assess_patient_factors"))

    history = env.state.tool_history
    ok = (
        len(history) == 3
        and all("jepa_info_gain" in h for h in history)
        and all("actual_delta" in h for h in history)
        and all("consistency_bonus" in h for h in history)
    )
    _log("full_episode_jepa_accumulation", ok,
         f"3 entries, gains={[h['jepa_info_gain'] for h in history]}")


# ---------------------------------------------------------------------------
# T11: JEPA world model loads without error
# ---------------------------------------------------------------------------
def test_world_model_loads():
    try:
        from env.environment import _get_world_model
        wm = _get_world_model()
        ok = wm is not None
        _log("world_model_loads", ok, f"model={type(wm).__name__}")
    except Exception as e:
        _log("world_model_loads", False, f"error: {e}")


# ---------------------------------------------------------------------------
# T12: env.step after done raises ValueError (not silently broken)
# ---------------------------------------------------------------------------
def test_step_after_done_raises():
    env = make_demo_env()
    env.step(AMRAction(action_type="COMMIT", prescription={
        "drug": "ceftazidime-avibactam", "dose": "1.25g IV q8h",
        "duration": "14 days", "justification": "test",
    }))
    try:
        env.step(AMRAction(action_type="INVESTIGATE",
                           tool_name="assess_patient_factors"))
        _log("step_after_done_raises", False, "expected ValueError not raised")
    except ValueError:
        _log("step_after_done_raises", True, "ValueError raised correctly")


# ---------------------------------------------------------------------------
# T13: Reward on correct prescription still > 0.5 with JEPA shaping active
# ---------------------------------------------------------------------------
def test_correct_rx_reward_with_jepa():
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
        "justification": "CRE K. pneumoniae IDSA first-line renal-adjusted",
    }))
    ok = obs.reward is not None and obs.reward >= 0.5
    _log("correct_rx_reward_with_jepa", ok,
         f"reward={obs.reward} (expect >= 0.5)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("AMR-Steward JEPA Integration Tests")
    print("=" * 60)

    tests = [
        test_action_to_jepa_key,
        test_jepa_fields_in_tool_history,
        test_jepa_info_gain_bounded,
        test_actual_delta_bounded,
        test_dense_cap_enforced,
        test_jepa_scaled_bonus_positive,
        test_repeated_tool_no_dense_bonus,
        test_consistency_bonus_bounded,
        test_unknown_tool_arg_graceful,
        test_full_episode_jepa_accumulation,
        test_world_model_loads,
        test_step_after_done_raises,
        test_correct_rx_reward_with_jepa,
    ]

    for fn in tests:
        try:
            fn()
        except Exception as e:
            _log(fn.__name__, False, f"EXCEPTION: {repr(e)}")

    print("=" * 60)
    if errors:
        print(f"FAILED ({len(errors)}): " + ", ".join(errors))
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED")
