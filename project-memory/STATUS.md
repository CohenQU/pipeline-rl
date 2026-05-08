# Status — pipeline-rl

Last updated: 2026-05-08

## Current Focus

Latent_thought reward sweep. Reward is `α·suffix_delta + β·joint_delta + γ·aux_delta − length_penalty`,
asserted `α+β+γ=1`. Two evaluator calls/rollout (treatment is split to derive aux_delta + conditional).
- **v00.x** (wikitext-103 train, wikitext-103 valid): 3×2 reward × penalty grid (v00.01–06), then γ sweep
  on suffix-only base (v00.07–09).
- **v01.x** (dolma3_dolmino streaming, train rows [0, 500K), test rows [500K, 501K)): mirrors selected
  v00 reward settings on the larger / more diverse corpus. v01.03, v01.05, v01.07, v01.08, v01.09 ready.
- All previous Slurm jobs (1596540/41/42 = v00.02/03/05) were scancelled at 2026-05-04 06:18 by an
  external SIGTERM after 14–22h of training. Checkpoints remain on disk.
- Currently nothing running. Nothing in queue.

## Waiting On

(no jobs running — all previous v00.x submissions scancelled 2026-05-04 06:18 after 14–22h)

Checkpoints from the cancelled runs are still on disk under
`/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v00.0{2,3,5}/finetune/intermediate/`.
With `wandb_resume: always` and `--requeue`, re-`sbatch`ing the v00 scripts resumes from the
last checkpoint.

## Pickup Instructions

1. Pick which configs to (re)submit. Available:
   - v00.x (wikitext): v00.02–v00.06 launched once and cancelled; v00.07–v00.09 (γ sweep) never launched.
   - v01.x (dolma streaming): v01.03, v01.05, v01.07, v01.08, v01.09 — never launched.
2. `sbatch /mnt/weka/home/wen.ye/workspace_m2/workspace/AI-Scientist/train/RL/bash/latent-thought-vXX.YY.sh`.
3. **First 5–15 min after a job starts**: watch for ERR-001 recurrence. In `<output_dir>/finetune/log/info_0.log`,
   "Batch queue is empty" should stop within ~2 min; in `<output_dir>/preprocess/info.log`, `published_samples`
   should reach 1024+ before any "Popped N old entries" lines. If popping starts and trainer never gets a
   batch — kill (don't wait for the NCCL timeout). See ERR-001.
4. Once past startup, watch wandb for first ~50 steps: `suffix_delta`, `joint_delta`, `aux_delta`, `aux_tokens`,
   `length_truncated`, `suffix_overlap_ratio` (proxy for copy-prefix pathology).
5. For v01.x: also watch the streaming dataset's first-row latency. Dolma is streamed from HF Hub, so the
   actor's first rollout will have an extra ~30–60s of dataset prefetch. After that it should be steady.
6. Plan: `docs/plans/2026-05-02-latent-thought-reward-redesign.md`. Reward code:
   `pipelinerl/domains/latent_thought/rollouts.py` (lines ~205–260). Preprocess gate:
   `pipelinerl/preprocess.py:551`. Loader: `pipelinerl/domains/latent_thought/load_datasets.py` (now
   supports `streaming`, `shuffle_seed`, `shuffle_buffer_size`, `skip_rows`, `max_rows` per spec).

## Do NOT Redo

- Existing v00.01–v00.06 yamls have been updated with explicit `reward_gamma: 0.0`. Do NOT remove.
- The static import test fails on login nodes (triton needs CUDA driver) — NOT a bug; use
  `python -c "import ast; ast.parse(open(...).read())"` for syntax checks.

## Key Context

See `decisions.md` (DEC-001..006), `hypotheses.md` (HYP-001..004), `experiments.md` (EXP-002..014),
`errors_and_fixes.md` (ERR-001).
