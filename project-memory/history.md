# Session History — pipeline-rl

<!-- Append-only log of work sessions. -->
<!-- Format: -->
<!-- ## YYYY-MM-DD Session N -->
<!-- - **Requests**: What the user asked for -->
<!-- - **Actions**: What was done -->
<!-- - **Discussion notes**: Key ideas explored, approaches considered, reasoning that led to decisions (especially for open-ended brainstorming sessions) -->
<!-- - **Outcome**: What was achieved -->
<!-- - **Artifacts produced**: Links to new entries in functions.md, decisions.md, hypotheses.md -->
<!-- - **Open items**: Anything left unfinished -->

## 2026-05-02 Session 1 — latent_thought reward redesign (v00.02–v00.06)
- **Requests**: User reported v00.01 reward hacking — policy generates overlong aux that copies prefix, gaming the joint-NLL denominator. Asked to redesign reward to measure suffix predictability given (prefix, aux), and add a length penalty for aux that hits max_tokens.
- **Actions**:
  - Mapped existing latent_thought rollout in `pipelinerl/domains/latent_thought/rollouts.py` (reward formula on lines 187–198, evaluator scoring via `evaluator_llm.get_batch_logprobs_token_ids`).
  - Designed hybrid reward `α · suffix_delta + β · joint_delta` with `α + β = 1` (DEC-001), and a hard length penalty (DEC-002).
  - Wrote plan to `docs/plans/2026-05-02-latent-thought-reward-redesign.md`.
  - Modified `pipelinerl/domains/latent_thought/rollouts.py`:
    - Updated module docstring with new reward formula and rationale.
    - Expanded `Metrics` class with `avg_nll_conditional`, `suffix_delta`, `joint_delta`, `reward_alpha_term`, `reward_beta_term`, `length_truncated`, `length_penalty_applied`, `aux_tokens_pre_trim`.
    - Added third `get_batch_logprobs_token_ids` call for `[prefix+aux] → [suffix]` (parallel via `asyncio.gather`).
    - Replaced reward computation; assert `α + β = 1`; subtract `length_penalty` when `output_length_tokens >= max_tokens`.
  - Created 5 new self-contained configs: `conf/latent-thought-v00.0{2,3,4,5,6}.yaml` (full copy of v00.01 with `latent_thought.{reward_alpha,reward_beta,length_penalty}` and `finetune.hub_model_revision` adjusted per the 3×2 grid).
  - Created 5 matching launch scripts: `train/RL/bash/latent-thought-v00.0{2,3,4,5,6}.sh` (copy of v00.01 with new `JOB_NAME` and `--config-name`).
  - Verified: AST parse OK, all 6 yaml configs parse with `α+β=1.0`, numerical regression confirms `α=0,β=1,lp=0` reproduces the v00.01 reward exactly.
- **Discussion notes**:
  - User initially said "we should also measure" — clarified whether to add or replace; settled on hybrid form so both behaviors are reachable from one code path.
  - Sign convention: `reward = baseline_NLL − conditional_NLL` so positive reward ⇔ aux made suffix easier (equivalent to the log-likelihood ratio `log p(suffix|prefix,aux) / p(suffix|prefix)`).
  - User briefly confused which axis of the hybrid maps to which old/new behavior; corrected mapping (α=0,β=1 = old; α=1,β=0 = new).
  - Considered renormalizing `joint_delta` to per-suffix-token scale before weighting; user accepted scale mismatch for v0 ("for now, that's fine").
  - Decided on hard length penalty (single scalar, default 0) rather than a soft ramp; cleaner ablation, can layer on later.
- **Outcome**: Code + 5 configs + 5 scripts + numerical regression test; ready to submit. v00.01 untouched and still maps to (α=0, β=1, lp=0) via code defaults.
- **Artifacts produced**: docs/plans/2026-05-02-latent-thought-reward-redesign.md; DEC-001, DEC-002, DEC-003; HYP-001, HYP-002, HYP-003; EXP-002 through EXP-006 queued.
- **Open items**: User to submit v00.03 first (cleanest "new" reward), then v00.02/04/05/06 once v00.03 looks healthy in wandb (project AI-CSI, tag `latent_thought`). Watch evaluator queue length (3 prompt_logprobs calls per rollout now, vs 2 in v00.01).

## 2026-05-02 Session 2 — diagnose preprocess deadlock, fix max_ready_samples_per_lead
- **Requests**: User reported v00.03 (Slurm 1595298) errored after running ~4h 24m. Asked to diagnose, fix, resubmit, and push.
- **Actions**:
  - Triaged the err log: NCCL `_REDUCE_SCATTER_BASE` watchdog timeout at SeqNum=475. That was the cascade, not the cause.
  - Cross-checked actor/preprocess/finetune logs:
    - Actor was healthy: 4688 train samples published.
    - Preprocessor wrote 544 samples to `training_data` (21:05:33–21:06:28) then started "Popped N old entries from processed entries queue" indefinitely.
    - Finetune trainer logged "Batch queue is empty" indefinitely; never took a step; eventually the idle NCCL collective hit the 10-min watchdog.
  - Read `pipelinerl/preprocess.py` and identified the gate at line 551: `published − processed > max_ready_samples_per_lead × num_trainers`. With base.yaml defaults (64 × 8 = 512) and `samples_per_step = train_batch_size(32) × gradient_accumulation_passes(32) = 1024`, the gate triggers before the trainer can take its first step → deadlock.
  - Verified by reading v00.00's finetune log: 8 "Starting training with 0 samples" entries before its first "Stop step at" — same race; v00.00 just got lucky on the 9th restart. v00.03 was more vulnerable because the new 3rd evaluator call slowed rollouts and tilted the race against publishing 1024 samples in one inner-loop pass.
  - Cancelled Slurm 1595298, 1595299, 1595300.
  - Patched `conf/latent-thought-v00.0{1,2,3,4,5,6}.yaml` to add `preprocess.max_ready_samples_per_lead: 256` (matches existing `rc_proof_qwen3-4b-*.yaml` pattern; gives gate threshold = 2048).
  - Resubmitted v00.03 (1596540), v00.05 (1596541), v00.02 (1596542).
  - Logged DEC-004 (the override decision), ERR-001 (the deadlock cause + diagnostic + prevention), updated EXP-002/003/005 with new job IDs.
- **Discussion notes**:
  - Initial suspicion was that the new 3rd evaluator call broke something in the rollout itself — ruled out by seeing 4688 healthy rollouts published.
  - Realized this was a pre-existing race (v00.00 hit it 8 times) only after grepping its finetune log for "Starting training with 0 samples".
  - Considered other fixes (raise to 1024, set `max_lag`, lower `gradient_accumulation_passes`); chose 256 because it matches a known-good production pattern (rc_proof configs).
- **Outcome**: All 3 jobs resubmitted with the fix; v00.03 v1 (1595298) wasted 4h 24m of cluster time before being killed.
- **Artifacts produced**: DEC-004, ERR-001; preprocess override now in all 6 latent-thought yamls.
- **Open items**: Watch the resubmitted jobs' first 5–15 minutes for ERR-001 recurrence (per STATUS.md Pickup Instructions). Once 1596540/1/2 take their first training step, watch the reward components in wandb.

