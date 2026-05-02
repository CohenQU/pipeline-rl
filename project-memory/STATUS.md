# Status — pipeline-rl

Last updated: 2026-05-02

## Current Focus

Sweeping latent_thought reward variants to fix v00.01 reward hacking (long aux that copies
prefix). Five new configs (v00.02–v00.06) cover a 3×2 grid of `(reward_alpha, reward_beta) ×
length_penalty`. Code, configs, and scripts are ready; nothing submitted yet.

## Waiting On

- **1595298** (PD, QOSGrpNodeLimit): latent-thought-v00.03 (α=1, β=0, lp=0). EXP-003.
  - Output: `/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v00.03/`
  - Logs: `/mnt/weka/home/wen.ye/workspace_m2/tmp/log/slurm/sft-1595298.out`
- **1595299** (PD, QOSGrpNodeLimit): latent-thought-v00.05 (α=0.5, β=0.5, lp=0). EXP-005.
  - Output: `/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v00.05/`
  - Logs: `/mnt/weka/home/wen.ye/workspace_m2/tmp/log/slurm/sft-1595299.out`
- **1595300** (PD, QOSGrpNodeLimit): latent-thought-v00.02 (α=0, β=1, lp=0.1). EXP-002.
  - Output: `/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v00.02/`
  - Logs: `/mnt/weka/home/wen.ye/workspace_m2/tmp/log/slurm/sft-1595300.out`

Check status: `squeue -j 1595298,1595299,1595300` or `sacct -j <id>`.
Wandb: project `AI-CSI`, tag `latent_thought`. Watch `suffix_delta`, `joint_delta`,
`aux_tokens`, `length_truncated`, evaluator queue.

## Pickup Instructions

1. If jobs are still pending, just wait — `QOSGrpNodeLimit` clears as cluster capacity frees up.
2. Once a job runs, watch wandb for first ~50 steps: `suffix_delta` and `joint_delta`
   populated and roughly the right sign, `aux_tokens` distribution dropping (no more
   pure copy-prefix behavior in v00.03/05), evaluator queue stable (3 prompt_logprobs
   calls/rollout now, was 2 in v00.01).
3. v00.04 and v00.06 are NOT submitted yet — user requested only 02/03/05.
4. Plan: `docs/plans/2026-05-02-latent-thought-reward-redesign.md`. Reward code:
   `pipelinerl/domains/latent_thought/rollouts.py` (~lines 200–260).

## Next Steps

After all 6 runs (v00.01 + v00.02–v00.06) have soaked, decide whether α=1 (suffix-only)
becomes the new default, and whether to keep `length_penalty=0.1`.

## Do NOT Redo

- Do NOT modify `conf/latent-thought-v00.01.yaml` — code defaults reproduce its reward exactly.
- Do NOT add a soft length-penalty ramp for v0 (DEC-002 chose hard penalty for clean A/B).
- The static import test fails on login nodes (triton needs CUDA driver) — NOT a bug; use
  `python -c "import ast; ast.parse(open(...).read())"` for syntax checks on login.

## Key Context

- Reward = `α · suffix_delta + β · joint_delta − length_penalty_applied`; `α+β=1` asserted.
- 3 evaluator `get_batch_logprobs_token_ids` calls/rollout (baseline, treatment, conditional).
- See `decisions.md` (DEC-001..003), `hypotheses.md` (HYP-001..003), `experiments.md` (EXP-002..006).
