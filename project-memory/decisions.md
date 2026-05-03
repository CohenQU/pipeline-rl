# Decisions — pipeline-rl

<!-- Format for each decision: -->
<!-- ## DEC-NNN: Short description -->
<!-- - **Date**: YYYY-MM-DD -->
<!-- - **Context**: What problem or question prompted this? -->
<!-- - **Decision**: What we chose -->
<!-- - **Alternatives considered**: What else was on the table -->
<!-- - **Reasoning**: Why this choice over the others -->

## DEC-001: Hybrid latent_thought reward (α·suffix_delta + β·joint_delta) with α+β=1
- **Date**: 2026-05-02
- **Context**: v00.01 latent_thought training showed reward hacking — policy emits long aux that copies prefix. Old reward `avg_NLL(suffix|prefix) − avg_NLL(aux⊕suffix|prefix)` divides treatment by `(|aux|+|suffix|)`, so easy aux tokens dilute the average and the suffix portion need not improve.
- **Decision**: Replace the reward with `α · suffix_delta + β · joint_delta` where `suffix_delta = avg_NLL(suffix|prefix) − avg_NLL(suffix|prefix,aux)` (both terms divided by `|suffix|`, immune to length dilution). Constrain `α + β = 1` (asserted in code). Defaults `α=0, β=1` reproduce the v00.01 reward exactly.
- **Alternatives considered**:
  - Pure suffix-only delta (no joint term) — clean but loses the variance-reduction benefit of joint scoring; we can recover it via α=1 within the hybrid form.
  - Margin form `relu(baseline − conditional)` — drops gradient on harmful aux; rejected for now to keep training signal symmetric.
  - Renormalize joint_delta to per-suffix-token scale before weighting — user accepted current scale mismatch ("for now, that's fine").
- **Reasoning**: Hybrid form subsumes both old and new behaviors via two scalars. The `α+β=1` constraint keeps reward magnitude comparable across configs in the sweep (v00.01–v00.06), and lets us isolate the contribution of the new suffix term cleanly.

## DEC-002: Hard length penalty (config scalar, default 0.0) when policy hits max_tokens
- **Date**: 2026-05-02
- **Context**: Reward-hack symptom included aux saturating the 1024-token cap. We need a direct disincentive for full-length aux.
- **Decision**: Add `latent_thought.length_penalty` config scalar. When `llm_call.output_length_tokens >= cfg.llm.parameters.max_tokens`, subtract `length_penalty` from the reward. Default `0.0` (off); compared at `0.1` in the sweep. Detection uses token count (LLMOutput in this stack does not carry `finish_reason`).
- **Alternatives considered**:
  - Soft ramp like `pipelinerl/domains/math/rollouts.py:50–56` (`length_penalty(max_length, sequence_length, buffer_tokens)`) — extra hyperparameter (buffer_tokens), and we want a clean A/B between "no penalty" and "penalty" first.
  - Both soft + hard — overcomplicates for v0.
- **Reasoning**: Single scalar gives a clean ablation, matches user's framing ("variable control the penalty, default is 0, 0.1 as comparison"). Can layer on a soft ramp later if the hard penalty is too brittle.

## DEC-004: Override `preprocess.max_ready_samples_per_lead` (64 → 256) in all latent_thought configs
- **Date**: 2026-05-02
- **Context**: First v00.03 attempt (Slurm 1595298) deadlocked after ~12 min. Trainer never received its first batch; NCCL collective then timed out at 10 min cascading the job failure. Investigation found preprocessor wrote 544/1024 samples needed for the trainer's first step, then stopped — gated at `published − processed > max_ready_samples_per_lead × num_trainers = 64 × 8 = 512`. With `samples_processed=0` (trainer never started), gate stayed shut. v00.00 had hit this too (8 restarts in its finetune log before lucky resolution). v00.03 was more vulnerable because 3 evaluator calls/rollout (vs 2 in v00.01) slowed the actor enough that the inner loop never published the full 1024 in one pass before the gate triggered.
- **Decision**: Add `preprocess.max_ready_samples_per_lead: 256` to all latent_thought configs (v00.01–v00.06). New gate threshold = `256 × 8 = 2048` (~2 steps in flight) — enough headroom for the 1024-per-step requirement.
- **Alternatives considered**:
  - Bump to a higher value (e.g. 1024) — more headroom but bigger memory footprint of in-flight rollouts; 256 matches the rc_proof configs that already work in production.
  - Set `max_lag` to disable `pop_old_data` and use a different gating policy — larger code-path change; not validated for this domain.
  - Reduce `gradient_accumulation_passes` to lower samples-per-step — would alter training dynamics, not safe to change just to dodge the deadlock.
- **Reasoning**: Aligns with the existing `rc_proof_qwen3-4b-*.yaml` pattern (same value, used in many shipped configs). Smallest possible change that removes the race.

## DEC-003: Full-copy versioning for latent_thought configs (no Hydra inheritance chain)
- **Date**: 2026-05-02
- **Context**: When creating v00.02–v00.06, we could either inherit from v00.01 via Hydra `defaults: [latent-thought-v00.01, _self_]` or fully copy v00.01.yaml and override specific fields.
- **Decision**: Each new yaml is a full copy of v00.01.yaml with only the relevant fields changed (`finetune.hub_model_revision`, `latent_thought.{reward_alpha,reward_beta,length_penalty}`).
- **Alternatives considered**: Hydra inheritance chain — DRYer but couples runs (a future edit to v00.01 cascades into all later configs).
- **Reasoning**: Matches the existing repo pattern (every prior `vNN.NN.yaml` is self-contained) and keeps each run reproducible from a single file.
