# Status — pipeline-rl

Last updated: 2026-05-09

## Current Focus

Latent_thought reward sweep. Reward is `α·suffix_delta + β·joint_delta + γ·aux_delta − length_penalty`,
asserted `α+β+γ=1`. Two evaluator calls/rollout (treatment is split to derive aux_delta + conditional).
- **v00.x** (wikitext-103): 3×2 reward × penalty grid (v00.01–06) + γ sweep on suffix-only base
  (v00.07–09). v00.02/03/05 ran 14–22h on 2026-05-03 then scancelled by external SIGTERM;
  checkpoints remain on disk.
- **v01.x** (dolma3_dolmino streaming via custom `jsonl_zst_hf` loader, train rows [0, 500K),
  test rows [500K, 501K)): mirrors selected v00 reward settings on the larger / more diverse
  corpus. **All 5 submitted on 2026-05-09**, currently PD on lowprio (preemptible).

## Waiting On

All 5 PD on lowprio (Priority preempted earlier, requeued via `--requeue`):

- **1608990** v01.03 (α=1.0, β=0.0, γ=0.0). EXP-010. Smoke test for v01 series.
  Already accumulated 1h 25m + 32 micro-batch steps + ckpt at step 25 before preempted.
- **1609112** v01.05 (α=0.5, β=0.5, γ=0.0). EXP-011.
- **1609113** v01.07 (α=0.9, β=0.0, γ=0.1). EXP-012.
- **1609114** v01.08 (α=0.7, β=0.0, γ=0.3). EXP-013.
- **1609115** v01.09 (α=0.5, β=0.0, γ=0.5). EXP-014.

Output: `/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v01.0X/`. Logs:
`/mnt/weka/home/wen.ye/workspace_m2/tmp/log/slurm/sft-160{8990,9112,9113,9114,9115}.{out,err}`.
Wandb project `latent-thought`. Status: `squeue --me`.

## Pickup Instructions

1. **Healthy startup signature** (~5–15 min after a job starts):
   - `<output_dir>/actor/error.log` stays 0 bytes (= ERR-002 not recurring)
   - `<output_dir>/preprocess/info.log` shows `Processed N samples ... wrote to training_data/0/0-8`
     (= dataset loading worked)
   - `<output_dir>/finetune/log/info_0.log` shows "Stop step at" entries (= training proceeding)
2. **Failure signatures**:
   - Actor `error.log` non-empty + pyarrow ArrowInvalid traceback → ERR-002 (dolma metadata
     schema drift) — kill and check loader path
   - "Batch queue is empty" indefinitely + "Popped N old entries" growing → ERR-001 (preprocess
     gate) — kill, don't wait for NCCL timeout
3. v00.04, v00.06, v00.07–v00.09 (γ sweep on wikitext) never launched. Bash scripts ready under
   `/mnt/weka/home/wen.ye/workspace_m2/workspace/AI-Scientist/train/RL/bash/latent-thought-v00.0X.sh`.
4. Plan: `docs/plans/2026-05-02-latent-thought-reward-redesign.md`. Code:
   `pipelinerl/domains/latent_thought/rollouts.py` (reward), `load_datasets.py` (now supports
   `loader_kind: jsonl_zst_hf` for `.jsonl.zst` HF datasets — bypasses pyarrow, see ERR-002).
5. All launch scripts now include `#SBATCH --qos=lowprio --partition=lowprio`. Lowprio jobs are
   preemptible; checkpoints + `wandb_resume: always` + `--requeue` mean they recover on requeue.

## Do NOT Redo

- Existing v00.01–v00.06 yamls have explicit `reward_gamma: 0.0`. Do NOT remove (needed for
  the α+β+γ=1 assert with the new reward formula).
- v01 yamls use `loader_kind: jsonl_zst_hf` — do NOT switch back to default `hf` for dolma; the
  pyarrow path crashes on cross-shard schema drift in `metadata` (ERR-002).
- The static import test fails on login nodes (triton needs CUDA driver) — NOT a bug; use
  `python -c "import ast; ast.parse(open(...).read())"` for syntax checks.

## Key Context

See `decisions.md` (DEC-001..006), `hypotheses.md` (HYP-001..004),
`experiments.md` (EXP-002..014), `errors_and_fixes.md` (ERR-001, ERR-002).
