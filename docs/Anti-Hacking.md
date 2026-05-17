# AMR-Steward: Anti-Hacking & Robustness Mechanisms

This document outlines the strict anti-hacking, anti-gaming, and robustness mechanisms engineered into the AMR-Steward reinforcement learning environment. 

The environment is designed to prevent reward hacking—a common failure mode in RL where the agent finds a loophole in the reward function to achieve high scores without actually solving the underlying task. 

Here is how the AMR-Steward reward pipeline guarantees safety, validity, and non-exploitable training signals:

---

## 1. Zero Subjectivity: Pure Python Deterministic Oracles
**Mechanism:** The entire reward pipeline (`env/reward.py`) is composed of pure Python functions performing strict lookups against established medical guidelines (EUCAST v16.0 breakpoints and IDSA guidelines).
**Anti-Hacking Property:** 
- **No LLM Judges:** There is no "LLM-as-a-judge" anywhere in the reward loop. 
- **No Prompt Injection:** The agent cannot use prompt injection or sycophancy to trick a judge into granting a higher score.
- **Zero Variance:** A specific prescription for a specific patient will yield the *exact* same reward every single time.

## 2. R0: The Absolute Allergy Safety Gate
**Mechanism:** `R0_allergy_safety` acts as an unbypassable hard gate.
**Anti-Hacking Property:** 
- If the agent prescribes a drug to which the patient is allergic, the entire reward calculation immediately short-circuits to **0.0**, regardless of how effective or narrow-spectrum the drug is.
- This prevents the model from trading off patient safety for "points" in other reward components. It must learn that safety is a hard constraint, not a soft penalty.

## 3. The Quality-Ratio Oracle (Patient-Specific Normalization)
**Mechanism:** The environment calculates an `opt_score` at the start of every episode by brute-forcing all drugs in the antibiogram to find the maximum possible `process_score` for *that specific patient*. The agent's reward is then scaled relative to this optimum: `quality_ratio = min(1.0, process_score / opt_score)`.
**Anti-Hacking Property:** 
- **Prevents Difficulty Exploitation:** An agent cannot inflate its average score by over-performing on "easy" cases. Every case is graded on a strict curve where 1.0 represents the best possible clinical decision for that exact scenario.
- **Invariant Reward Ceiling:** The theoretical maximum is always tightly bounded.

## 4. Conditional Gating on Stewardship (R3 dependent on R1)
**Mechanism:** The stewardship reward (`R3_stewardship`), which rewards the use of the narrowest-spectrum drug, contains a conditional check: `if r1_score == 0.0: return 0.0`.
**Anti-Hacking Property:** 
- **Prevents "Do-Nothing" Hacking:** Without this gate, an agent could hack the stewardship score by prescribing the weakest, narrowest-spectrum drug available (like Penicillin) even if the bacteria is completely resistant to it.
- The agent only receives stewardship points if the drug *actually works* against the pathogen.

## 5. R5 Tool Efficiency: Anti-Spam & Anti-Spoofing
**Mechanism:** The `R5_tool_efficiency` reward evaluates the agent's investigation behavior. It scores based on `unique_tool_types / max(1, budget_spent)`.
**Anti-Hacking Property:** 
- **Anti-Spam:** By counting unique tool *types* (rather than raw tool calls), the agent is penalized for spamming the same tool repeatedly just to exhaust its budget or artificially inflate its "investigation" score.
- **Anti-Spoofing (State-Backed Logging):** The R5 score operates on a structured `{tool, arg}` log stored in `AMRState.tool_history`. It does not use regex text parsing to count tools. This means the LLM cannot hallucinate or "spoof" fake tool call logs in its final output to trick the reward function into thinking it investigated.

## 6. Strict Output Formatting (R6)
**Mechanism:** The `R6_format` reward requires a single, clean `COMMIT` line. It starts decaying the score by `0.05` for every line over 3 lines.
**Anti-Hacking Property:** 
- **Prevents "Shotgunning":** The agent cannot output 50 different prescriptions and hope the parser picks up the right one. It is forced to be concise, decisive, and strictly adhere to the expected operational format.

---

### Conclusion for Evaluators
AMR-Steward provides a watertight RL environment. By separating the reward logic from the LLM, implementing hard clinical safety gates, normalizing scores dynamically per-patient, and securing the tool-call tracking, the environment guarantees that **the only way to maximize the reward is to learn genuine, IDSA-concordant antimicrobial stewardship.**
