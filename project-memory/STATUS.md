# Status — pipeline-rl

Last updated: 2026-05-08

## Current Focus

Latent_thought reward sweep. Reward is `α·suffix_delta + β·joint_delta + γ·aux_delta − length_penalty`,
asserted `α+β+γ=1`. Two evaluator calls/rollout (treatment is split to derive aux_delta + conditional).
- v00.01–v00.06: 3×2 grid of `(α,β) × length_penalty`.
- v00.07–v00.09: γ sweep (0.1, 0.3, 0.5) on suffix-only base; β=0, no length_penalty.
- Submitted: v00.02, v00.03, v00.05. Not yet: v00.04, v00.06, v00.07, v00.08, v00.09.

## Waiting On

Resubmitted after preprocess gate fix (DEC-004 / ERR-001). The earlier batch (1595298–1595300) cancelled
because v00.03 deadlocked on the `max_ready_samples_per_lead` race.

- **1596540** PD: latent-thought-v00.03 (α=1, β=0, γ=0, lp=0). EXP-003. Logs sft-1596540.{out,err}.
- **1596541** PD: latent-thought-v00.05 (α=0.5, β=0.5, γ=0, lp=0). EXP-005. Logs sft-1596541.{out,err}.
- **1596542** PD: latent-thought-v00.02 (α=0, β=1, γ=0, lp=0.1). EXP-002. Logs sft-1596542.{out,err}.

Output: `/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v00.0X/`. Log dir:
`/mnt/weka/home/wen.ye/workspace_m2/tmp/log/slurm/`. Status: `squeue -j 1596540,1596541,1596542`.
Wandb: project `AI-CSI`, tag `latent_thought`.

## Pickup Instructions

1. **First 5–15 min after a job starts**: watch for ERR-001 recurrence. In `<output_dir>/finetune/log/info_0.log`,
   "Batch queue is empty" should stop within ~2 min; in `<output_dir>/preprocess/info.log`, `published_samples`
   should reach 1024+ before any "Popped N old entries" lines. If popping starts and trainer never gets a
   batch — kill (don't wait for the NCCL timeout). See ERR-001.
2. Once past startup, watch wandb for first ~50 steps: `suffix_delta`, `joint_delta`, `aux_delta`, `aux_tokens`,
   `length_truncated`, `suffix_overlap_ratio` (proxy for copy-prefix pathology).
3. Submit γ sweep (v00.07/08/09) once v00.03 is healthy and we have a baseline curve.
4. Plan: `docs/plans/2026-05-02-latent-thought-reward-redesign.md`. Reward code:
   `pipelinerl/domains/latent_thought/rollouts.py` (lines ~205–260). Preprocess gate:
   `pipelinerl/preprocess.py:551`.

## Do NOT Redo

- Existing v00.01–v00.06 yamls have been updated with explicit `reward_gamma: 0.0`. Do NOT remove.
- The static import test fails on login nodes (triton needs CUDA driver) — NOT a bug; use
  `python -c "import ast; ast.parse(open(...).read())"` for syntax checks.

## Key Context

See `decisions.md` (DEC-001..005), `hypotheses.md` (HYP-001..004), `experiments.md` (EXP-002..009),
`errors_and_fixes.md` (ERR-001).
