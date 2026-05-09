# Errors & Fixes — pipeline-rl

<!-- Log non-trivial errors and their resolutions so we don't re-debug them. -->
<!-- Format: -->
<!-- ## ERR-NNN: Short description -->
<!-- - **Date**: YYYY-MM-DD -->
<!-- - **Error**: The error message or symptom -->
<!-- - **Cause**: Root cause -->
<!-- - **Fix**: What resolved it -->
<!-- - **Prevention**: How to avoid in the future (if applicable) -->

## ERR-002: dolma3_dolmino streaming load crashes actor with pyarrow JSON parse error
- **Date**: 2026-05-09
- **Error**: First v01.03 run (job 1607796) hung at startup; logs revealed actor crashed in `dataset_loader(cfg.test_dataset_names, ...)` with `pyarrow.lib.ArrowInvalid: JSON parse error: Column(/metadata/google_gemma-3-12b-it_contains_pii/[]/[]) changed from number to boolean in row 0`. Trainer kept waiting for actor data forever (no NCCL timeout in this case because trainer hadn't started training yet — it was still in the "waiting for actor/0/0 to be created" loop). Job ran 13+ hours doing nothing before being noticed.
- **Cause**: The `allenai/dolma3_dolmino_mix-10B-1025` dataset has cross-shard schema drift in its `metadata` column. Different shards encode the same nested field (`metadata/google_gemma-3-12b-it_contains_pii`) with different types — number in some, boolean in others. HF's JSON builder uses pyarrow's `read_json` to parse each shard, which infers a per-shard schema. When it can't reconcile types, it raises ArrowInvalid. The crash occurred even with `streaming=True` because pyarrow does the per-row schema check at parse time.
- **Fix**: Project columns BEFORE iteration via `dataset.select_columns(["text"])`. HF pushes this down into the JSON reader so pyarrow only materializes the requested columns, sidestepping the bad `metadata` column entirely. Implemented as `keep_columns: list[str]` per-spec option in `load_datasets.py`, applied after `load_dataset()` and before any shuffle/skip/take. All five v01 dolma yamls now set `keep_columns: ["text"]`.
- **Prevention**:
  1. For any large multi-shard HF dataset (especially CommonCrawl-derived ones with rich metadata), set `keep_columns` to ONLY the fields you actually consume.
  2. Symptom signature: actor's `error.log` is non-empty and `info.log` shows `dataset_loader(...test_dataset_names...)` traceback; preprocessor stuck on "Waiting for actor/0/0 to be created" indefinitely. Kill the job — the deadlock won't resolve.

## ERR-001: latent_thought RL job hangs on first step then NCCL-times out
- **Date**: 2026-05-02
- **Error**: Trainer logs `Batch queue is empty, retrying` indefinitely; ~10 min later NCCL collective `WorkNCCL(SeqNum=475, OpType=_REDUCE_SCATTER_BASE)` times out across all finetune ranks; the whole multi-node job aborts. Looks like a network/NCCL bug but is actually a data-pipeline deadlock. Symptom seen on Slurm 1595298 (v00.03), wasted ~4h 24m before cancellation.
- **Cause**: The preprocessor's gate at `pipelinerl/preprocess.py:551` skips publishing when `published_samples − trainer_state.samples_processed > max_ready_samples_per_lead × num_trainers`. With `base.yaml` defaults (`max_ready_samples_per_lead: 64`, `num_trainers = total_finetune_gpus = 8`), the threshold is 512. The trainer needs `train_batch_size × gradient_accumulation_passes = 1024` samples to take its first step; until it does, `samples_processed = 0`. So once the preprocessor publishes ~512–544 samples, the gate stays shut forever and the trainer cannot get the remaining ~480 to start. The trainer eventually crashes when its idle background NCCL collective hits its 10-min watchdog timeout. Intermittent — depends on whether the preprocessor's inner loop happens to publish 1024 in one outer-loop pass before the gate triggers; v00.00 hit this 8 times before a lucky restart got past it. v00.03 was more vulnerable because the rollout function does 3 evaluator `get_batch_logprobs_token_ids` calls (vs 2 in v00.01), slowing the actor and making the race tilt against publishing 1024 in one pass.
- **Fix**: Override `preprocess.max_ready_samples_per_lead: 256` in each latent-thought config (gives gate threshold = 256 × 8 = 2048, ~2 steps in flight). Matches the value used in `rc_proof_qwen3-4b-*.yaml`. See DEC-004.
- **Prevention**:
  1. New domain configs that use `train_batch_size × gradient_accumulation_passes ≥ 512` MUST override `max_ready_samples_per_lead` so that `value × total_finetune_gpus > samples_per_global_step`.
  2. If a job's finetune log shows "Batch queue is empty" for more than 1 minute combined with the preprocessor showing "Popped N old entries from processed entries queue" growing, kill it — the deadlock will not resolve and you will lose the cluster time waiting for the NCCL watchdog.
  3. Diagnostic: check `published_samples` vs `samples_processed` in the preprocess info.log. If published stalls < `samples_per_step` while popping increases, this is the gate.
