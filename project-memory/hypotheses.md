# Hypotheses — pipeline-rl

<!-- Research hypotheses and TODOs that guide what to try next. -->
<!-- These help agents reason about *why* to run an experiment, not just *what* to run. -->

<!-- Format: -->
<!-- ## HYP-NNN: Short statement -->
<!-- - **Status**: open | testing (EXP-NNN) | supported | refuted | abandoned -->
<!-- - **Reasoning**: Why we think this might be true -->
<!-- - **How to test**: What experiment or analysis would confirm/refute this -->
<!-- - **Evidence so far**: Any partial results or observations -->

## HYP-001: The v00.01 reward hack is caused by joint-NLL denominator dilution
- **Status**: testing (EXP-002, EXP-003)
- **Reasoning**: v00.01 reward = `avg_NLL(suffix|prefix) − avg_NLL(aux⊕suffix|prefix)`. The treatment term divides by `(|aux|+|suffix|)`, so any aux that the evaluator finds easy (e.g., a copy of prefix) drags the mean down without making suffix any easier. Switching the second term to `avg_NLL(suffix|prefix,aux)` (divide by `|suffix|` only) removes the lever.
- **How to test**: Compare aux length distribution and `suffix_overlap_ratio` in v00.03 (α=1, β=0, no penalty) vs v00.01. If the hack is denominator-driven, v00.03 should show shorter aux and lower overlap with prefix, with positive `suffix_delta` even when `joint_delta` is small.
- **Evidence so far**: Qualitative — user reported aux often saturates the 1024 cap with prefix-like content in v00.01.

## HYP-002: A hard length penalty (0.1) at max_tokens further shortens aux without reducing predictive lift
- **Status**: testing (EXP-004, EXP-006)
- **Reasoning**: Even with the suffix-only reward, the policy may still emit unnecessarily long aux because there is no incentive to be terse. A small constant penalty applied only when the cap is hit nudges away from the saturation regime without distorting the gradient elsewhere.
- **How to test**: Compare `aux_tokens` distribution and final `suffix_delta` in v00.04 (α=1, β=0, lp=0.1) vs v00.03 (lp=0). If HYP-002 holds, aux_tokens.mean should drop while `suffix_delta` stays comparable.
- **Evidence so far**: None.

## HYP-003: A 50/50 hybrid produces a smoother training signal than either term alone
- **Status**: testing (EXP-005, EXP-007)
- **Reasoning**: `joint_delta` has lower variance because it averages over more tokens, but is hackable. `suffix_delta` is unhackable but only sees |suffix| tokens of signal per rollout. A blend may reduce gradient noise without reintroducing the dilution hack at full strength.
- **How to test**: Compare wall-clock convergence of `reward` and `suffix_delta` between v00.03 (α=1) and v00.05 (α=0.5). If HYP-003 holds, v00.05 reaches the same `suffix_delta` faster but both terms remain bounded (no copy-prefix regression).
- **Evidence so far**: None. Note: the two terms have different denominators, so equal weights ≠ equal magnitude (joint_delta typically smaller). Decision to leave unnormalized was deliberate (DEC-001).
