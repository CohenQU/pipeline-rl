# Errors & Fixes — pipeline-rl

<!-- Log non-trivial errors and their resolutions so we don't re-debug them. -->
<!-- Format: -->
<!-- ## ERR-NNN: Short description -->
<!-- - **Date**: YYYY-MM-DD -->
<!-- - **Error**: The error message or symptom -->
<!-- - **Cause**: Root cause -->
<!-- - **Fix**: What resolved it -->
<!-- - **Prevention**: How to avoid in the future (if applicable) -->

## ERR-001: latent_thought RL job hangs on first step then NCCL-times out
- **Date**: 2026-05-02
- **Error**: Trainer logs `Batch queue is empty, retrying` indefinitely; ~10 min later NCCL collective `WorkNCCL(SeqNum=475, OpType=_REDUCE_SCATTER_BASE)` times out across all finetune ranks; the whole multi-node job aborts. Looks like a network/NCCL bug but is actually a data-pipeline deadlock. Symptom seen on Slurm 1595298 (v00.03), wasted ~4h 24m before cancellation.
- **Cause**: The preprocessor's gate at `pipelinerl/preprocess.py:551` skips publishing when `published_samples − trainer_state.samples_processed > max_ready_samples_per_lead × num_trainers`. With `base.yaml` defaults (`max_ready_samples_per_lead: 64`, `num_trainers = total_finetune_gpus = 8`), the threshold is 512. The trainer needs `train_batch_size × gradient_accumulation_passes = 1024` samples to take its first step; until it does, `samples_processed = 0`. So once the preprocessor publishes ~512–544 samples, the gate stays shut forever and the trainer cannot get the remaining ~480 to start. The trainer eventually crashes when its idle background NCCL collective hits its 10-min watchdog timeout. Intermittent — depends on whether the preprocessor's inner loop happens to publish 1024 in one outer-loop pass before the gate triggers; v00.00 hit this 8 times before a lucky restart got past it. v00.03 was more vulnerable because the rollout function does 3 evaluator `get_batch_logprobs_token_ids` calls (vs 2 in v00.01), slowing the actor and making the race tilt against publishing 1024 in one pass.
- **Fix**: Override `preprocess.max_ready_samples_per_lead: 256` in each latent-thought config (gives gate threshold = 256 × 8 = 2048, ~2 steps in flight). Matches the value used in `rc_proof_qwen3-4b-*.yaml`. See DEC-004.
- **Prevention**:
  1. New domain configs that use `train_batch_size × gradient_accumulation_passes ≥ 512` MUST override `max_ready_samples_per_lead` so that `value × total_finetune_gpus > samples_per_global_step`.
  2. If a job's finetune log shows "Batch queue is empty" for more than 1 minute combined with the preprocessor showing "Popped N old entries from processed entries queue" growing, kill it — the deadlock will not resolve and you will lose the cluster time waiting for the NCCL watchdog.
  3. Diagnostic: check `published_samples` vs `samples_processed` in the preprocess info.log. If published stalls < `samples_per_step` while popping increases, this is the gate.
