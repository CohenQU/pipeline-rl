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

## 2026-05-08 Session 1 — add aux_delta term + drop conditional eval call
- **Requests**: User wants a third reward term on `aux | prefix` so we optimize both "aux is fluent given prefix" and "suffix is predictable given prefix+aux". After exploring options, settled on the delta-form `aux_delta = avg_NLL(suffix|prefix) − avg_NLL(aux|prefix)` weighted by γ, with α+β+γ=1.
- **Discussion notes**:
  - Realized the existing `treatment` evaluator call (per-token logprobs of `aux⊕suffix | prefix`) already contains `log p(aux | prefix)` (first |aux| tokens) and `log p(suffix | prefix, aux)` (last |suffix| tokens). So splitting the treatment result gives us BOTH the new aux term AND the existing conditional NLL for free — we can drop the third evaluator call (which we added in session 1 today) and net 2 calls/rollout instead of 3 (back to v00.01 cost).
  - Flagged that γ > 0 has a built-in tension with α: aux being more predictable (γ wants this) is the same direction as the v00.01 copy-prefix hack. So small γ ≈ fluency regularizer, large γ ≈ re-introduces the pathology. User picked a γ sweep (0.1, 0.3, 0.5) on top of suffix-only base (β=0, varying α) to find the sweet spot.
  - Decided to default `reward_gamma: 0.0` so v00.01–v00.06 still pass the new α+β+γ=1 assert without any other edits, and added the explicit knob to all existing configs for self-documentation.
- **Actions**:
  - Refactored `pipelinerl/domains/latent_thought/rollouts.py`:
    - Updated `_sum_logprobs(result, start, end)` to support slicing.
    - Dropped the third `get_batch_logprobs_token_ids([prefix+aux], [suffix])` call.
    - Split treatment result into aux portion and suffix portion.
    - Added `avg_nll_aux`, `aux_delta`, `reward_gamma`, `reward_gamma_term` to the reward path and `Metrics` class.
    - Updated assert from `α+β=1` to `α+β+γ=1`.
    - Rewrote module docstring.
  - Added `reward_gamma: 0.0` line to `conf/latent-thought-v00.0{1,2,3,4,5,6}.yaml` for explicit documentation.
  - Created `conf/latent-thought-v00.0{7,8,9}.yaml` and matching launch scripts for the γ sweep on top of suffix-only base: (α=0.9, γ=0.1), (α=0.7, γ=0.3), (α=0.5, γ=0.5). All β=0, no length penalty.
  - Numerical regression: 6 unit checks pass. The treatment-split conditional matches what a separate conditional call would return (T5).
- **Outcome**: Code, configs, scripts ready. Same evaluator cost as v00.01 (2 calls/rollout). All 10 latent-thought yamls verified to sum to 1.0 across (α,β,γ).
- **Artifacts produced**: DEC-005 (the aux_delta + treatment-split decision), HYP-004 (γ sweep hypothesis), EXP-007/008/009 (three γ sweep entries).
- **Open items**: NOT submitted to Slurm — the in-flight v00.02/03/05 (1596540–1596542) have priority. User will likely submit the γ sweep after seeing initial v00.03 results.

## 2026-05-08 Session 2 — v01 series on dolma3_dolmino
- **Requests**: User asked to create v01.07/08/09 (mirroring v00.07/08/09 reward) trained on `allenai/dolma3_dolmino_mix-10B-1025`. Then asked to also create v01.03 and v01.05 (mirroring v00.03 and v00.05). Specified test set should be a 1K random subset from the train split of dolmino (not the wikitext val I initially proposed).
- **Discussion notes**:
  - User mentioned dolma has columns `id` and `text` — the loader's `text_field: text` already matches; `id` is unused.
  - Dolma at 10B tokens cannot be fully materialized; needed streaming + max_rows to subsample.
  - For "1K random test from train split": settled on the streaming idiom of "same shuffle_seed in both train and test specs, then `skip_rows` in test". Same seed → same shuffled stream order; train takes [0, 500K), test takes [500K, 501K). Cleanly disjoint and deterministic.
  - Also checked Slurm status — all three v00 jobs (1596540/41/42) had been scancelled at 2026-05-04 06:18 by an external SIGTERM after 14–22h of training; checkpoints remain on disk.
- **Actions**:
  - Extended `pipelinerl/domains/latent_thought/load_datasets.py` to accept per-spec `streaming`, `shuffle_seed`, `shuffle_buffer_size`, `skip_rows`, `max_rows`. Operations apply in order: load → shuffle → skip → take/select.
  - Created `conf/latent-thought-v01.0{3,5,7,8,9}.yaml`. v01.03 and v01.05 were sed-copied from v01.07 then header rewritten; all five share identical dolma streaming config (shuffle_seed=42, train max_rows=500000, test skip_rows=500000 max_rows=1000) and differ only in the (α, β, γ) tuple.
  - Created matching launch scripts.
  - Verified all 5 v01 yamls parse, sum to 1.0, and have matching train/test slices.
- **Outcome**: Loader changes are backward-compatible (defaults preserve old behavior). Five new dolma configs and launch scripts ready to submit. Nothing in queue.
- **Artifacts produced**: DEC-006, EXP-010 through EXP-014.
- **Open items**: User to decide whether to submit v00.x and/or v01.x jobs. Suggest submitting v01.03 first (suffix-only smoke run on dolma) to verify the streaming dataset works end-to-end, then queue the others.

## 2026-05-09 Session 1 — debug + fix dolma loader, launch full v01 sweep
- **Requests**: User submitted v01.03 (1607783, then 1607796 after wandb rename), then submitted v01.07 manually (1608942). All hung 13+h with empty actor `error.log` written but no progress. User asked to diagnose, fix, push, and resubmit.
- **Actions / discussion notes**:
  - Diagnosed: actor crashed at startup loading the dolma test dataset with `pyarrow.lib.ArrowInvalid: JSON parse error: Column(/metadata/google_gemma-3-12b-it_contains_pii/[]/[]) changed from number to boolean in row 0`. The bad shard is `science_tech_p050_shard_00000466.jsonl.zst`. Trainer waited indefinitely for actor data (no NCCL timeout because trainer hadn't even started training).
  - First fix attempt: `keep_columns: ["text"]` via `dataset.select_columns`. A 2000-row offline test passed and convinced me it worked — but that was sampling luck (the bad shard wasn't in the first 2000). Reproducing the EXACT loader path (load → select_columns → shuffle → skip → take) at higher row counts hit the same ArrowInvalid. **Lesson**: HF's `select_columns` does NOT push down to pyarrow's JSON parser, so it cannot rescue datasets with cross-shard schema drift. Stress-test column-projection workarounds before believing them.
  - Real fix: bypass HF + pyarrow entirely via custom `_stream_jsonl_zst_from_hub` loader — `HfFileSystem` enumerates `.jsonl.zst` shards, `zstandard.ZstdDecompressor().stream_reader` decompresses each, stdlib `json.loads` parses each line (per-row, tolerates schema drift). New `loader_kind: jsonl_zst_hf` per-spec option triggers this path; default `loader_kind: hf` preserves existing wikitext behavior. Verified offline: 50 train + 10 test rows from same `shuffle_seed` are perfectly disjoint.
  - Side investigation: confirmed editable pipelinerl install points at imo-dev fork (which lacks `latent_thought` domain), but Python finds AI-Scientist's local `pipelinerl/` via cwd-in-sys.path during launch. So my edits to AI-Scientist DO get loaded. (Still a fragile setup worth fixing later.)
  - Added `#SBATCH --qos=lowprio --partition=lowprio` to all 15 latent-thought launch scripts (only v01.07.sh had it before). Lowprio jobs are preemptible but get re-queued via `--requeue`; combined with `wandb_resume: always` and `save_checkpoint_steps: 25`, this is acceptable.
  - Submitted v01.03 (1608990). Verified it ran cleanly: 33,160 actor samples, 32 micro-batch steps, ckpt at step 25 in 1h 25m before getting preempted (and re-queued). Actor `error.log` stayed empty.
  - Submitted v01.05/07/08/09 (1609112/3/4/5). All PD on lowprio Priority.
  - Bumped `save_checkpoint_steps` from 10 to 25 across all 14 active configs (v00.00 untouched).
  - Renamed wandb project `AI-CSI` → `latent-thought` across all 15 latent-thought yamls. Cancelled the previous v01.03 (1607783) and resubmitted (1607796) so it would log to the renamed project.
  - Pushed multiple commits to `git@github.com.q:CohenQU/pipeline-rl.git` aicsi: 21a22d4 (rename), 89af1eb (failed keep_columns "fix"), 7429c58 (memory), 988dc96 (real custom-loader fix).
- **Outcome**: ERR-002 actually fixed. v01 sweep is in flight (5 jobs PD on lowprio), v01.03 already proven to train cleanly. v00 sweep partially complete; v00.04/06/07/08/09 never launched.
- **Artifacts produced**: ERR-002 (rewritten with real fix + lessons learned), EXP-010..014 updated with new job IDs and progress.
- **Open items**: Watch the 5 v01 jobs through their first checkpoints; compare suffix_delta / aux_delta / aux_tokens / suffix_overlap_ratio across (α, γ) settings on dolma. Eventually launch v00.04/06/07/08/09 too, OR decide v01 is the canonical sweep and skip them.

