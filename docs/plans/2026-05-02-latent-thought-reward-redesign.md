# Latent-Thought v00.02–v00.06: Reward Redesign + Length Penalty

Date: 2026-05-02

## Context

The v00.01 latent_thought RL run (`conf/latent-thought-v00.01.yaml`) exhibited reward hacking:
the policy generates very long AUX that essentially copies PREFIX. Because the current reward is

```
reward_v01 = avg_NLL(suffix | prefix) − avg_NLL(aux ⊕ suffix | prefix)
                                          └──── divided by (|aux| + |suffix|) ────┘
```

inflating AUX with easy-to-predict tokens (e.g. PREFIX copies) shrinks the *joint* average NLL
without actually making SUFFIX easier to predict. The denominator dilution is the hack.

We will sweep two design changes and compare against v00.01:

1. **Hybrid reward** weighting a new "suffix-only" delta (clean signal, both terms divided by
   `|suffix|`) against the existing joint delta:
   ```
   suffix_delta = avg_NLL(suffix | prefix) − avg_NLL(suffix | prefix, aux)
   joint_delta  = avg_NLL(suffix | prefix) − avg_NLL(aux ⊕ suffix | prefix)
   reward       = α · suffix_delta + β · joint_delta − length_penalty_term
   ```
   with the constraint `α + β = 1` (asserted in code).

2. **Hard length penalty** subtracted from reward when aux generation hit `max_tokens`
   (i.e. truncated by length, not natural EOS). Configurable scalar, default `0.0` (off),
   compared at `0.1`.

Six configurations total cover the 3×2 grid:

| Config        | α   | β   | length_penalty | Notes                   |
|---------------|-----|-----|----------------|-------------------------|
| v00.01 (kept) | 0.0 | 1.0 | 0.0            | existing baseline       |
| v00.02 (new)  | 0.0 | 1.0 | 0.1            | old reward + penalty    |
| v00.03 (new)  | 1.0 | 0.0 | 0.0            | suffix-only             |
| v00.04 (new)  | 1.0 | 0.0 | 0.1            | suffix-only + penalty   |
| v00.05 (new)  | 0.5 | 0.5 | 0.0            | hybrid                  |
| v00.06 (new)  | 0.5 | 0.5 | 0.1            | hybrid + penalty        |

## Implementation

### 1. `pipelinerl/domains/latent_thought/rollouts.py`

**Module docstring (top):** rewrite the "Reward" section to describe the hybrid form and the
length-penalty subtraction.

**`Metrics` class (lines 39–47):** expand so wandb shows every reward component.
Existing kept as-is: `avg_nll_baseline`, `avg_nll_treatment`, `nll_delta`, `aux_tokens`,
`prefix_tokens`, `suffix_tokens`, `cut_offset`, `suffix_overlap_ratio`. Add:

- `avg_nll_conditional: float = 0.0` — `avg_NLL(suffix | prefix, aux)`, the new third term
- `suffix_delta: float = 0.0` — `avg_nll_baseline − avg_nll_conditional`
- `joint_delta: float = 0.0` — `avg_nll_baseline − avg_nll_treatment` (same value as the
  existing `nll_delta`; keep both so v00.01 dashboards continue to work and the new field
  has a self-explanatory name)
- `reward_alpha_term: float = 0.0` — `α · suffix_delta`, the per-rollout contribution from α
- `reward_beta_term: float = 0.0` — `β · joint_delta`, the per-rollout contribution from β
- `length_truncated: bool = False` — True when generation hit `max_tokens`
- `length_penalty_applied: float = 0.0` — the actual scalar subtracted (0.0 when off or
  not truncated; equals `length_penalty` value when triggered)
- `aux_tokens_pre_trim: int = 0` — aux token count BEFORE the `max_total_tokens` budget
  trim (lines 170–172). Useful for diagnosing how often the budget trim fires vs. raw
  policy generation length.

Together with the existing `reward` field on `BaseMetrics` (and the `reward` set on
`trace.reward`), this gives wandb the full decomposition:
`reward = α · suffix_delta + β · joint_delta − length_penalty_applied`.

**Evaluator calls (rollouts.py lines 174–186):** replace the existing two-call gather with three
parallel calls:
```python
baseline_result, treatment_result, conditional_result = await asyncio.gather(
    asyncio.to_thread(evaluator_llm.get_batch_logprobs_token_ids,
                      [prefix_ids], [suffix_ids]),
    asyncio.to_thread(evaluator_llm.get_batch_logprobs_token_ids,
                      [prefix_ids], [aux_ids + suffix_ids]),
    asyncio.to_thread(evaluator_llm.get_batch_logprobs_token_ids,
                      [prefix_ids + aux_ids], [suffix_ids]),
)
```
The third call adds throughput pressure on the evaluator — note in the `evaluator_vllm_config`
docstring that we now do 3 prompt_logprobs calls per rollout. The evaluator config in v00.01
already runs at `max-num-seqs: 2` and chunked prefill, so this should be OK; flag in the plan
that we should monitor evaluator queue length on the first run.

**Reward computation (lines 187–198):** replace with
```python
sum_lp_baseline    = _sum_logprobs(baseline_result[0])
sum_lp_treatment   = _sum_logprobs(treatment_result[0])
sum_lp_conditional = _sum_logprobs(conditional_result[0])

n_suffix = max(1, len(suffix_ids))
n_joint  = max(1, len(aux_ids) + len(suffix_ids))

avg_nll_baseline    = -sum_lp_baseline    / n_suffix
avg_nll_treatment   = -sum_lp_treatment   / n_joint
avg_nll_conditional = -sum_lp_conditional / n_suffix

suffix_delta = avg_nll_baseline - avg_nll_conditional
joint_delta  = avg_nll_baseline - avg_nll_treatment

alpha = float(lt_cfg.get("reward_alpha", 0.0))
beta  = float(lt_cfg.get("reward_beta",  1.0))
assert abs(alpha + beta - 1.0) < 1e-6, f"reward_alpha + reward_beta must equal 1.0, got {alpha + beta}"

reward = alpha * suffix_delta + beta * joint_delta
```

**Length penalty:** detect hit-cap and subtract.
```python
length_penalty_value = float(lt_cfg.get("length_penalty", 0.0))
# Prefer llm_call.output.finish_reason == "length" if exposed by the LLMOutput;
# fall back to comparing the policy-side aux token count to llm.parameters.max_tokens.
max_tokens_cap = int(cfg.llm.parameters.max_tokens)
length_truncated = (
    getattr(llm_call.output, "finish_reason", None) == "length"
    or llm_call.output_length_tokens >= max_tokens_cap
)
length_penalty_applied = length_penalty_value if length_truncated else 0.0
reward = reward - length_penalty_applied
```

Verify finish_reason exposure: `grep -rn "finish_reason" pipelinerl/ tapeagents/`. If
`LLMOutput` does not carry it, the `output_length_tokens >= max_tokens_cap` fallback is
sufficient (math domain uses an analogous pattern at `pipelinerl/domains/math/rollouts.py:50–56`).

**Discount (existing line 196–198):** keep, applied after the penalty.

**Metrics population (lines 211–224):** populate every new field listed in the `Metrics`
section above, keeping the existing fields and setting `nll_delta=joint_delta` for
backward compatibility with v00.01 dashboards.

```python
metrics = Metrics(
    reward=float(reward),
    success=reward > 0.0,
    no_error=True,
    no_answer=not bool(aux_text.strip()),
    # NLL terms (running monitors)
    avg_nll_baseline=float(avg_nll_baseline),
    avg_nll_treatment=float(avg_nll_treatment),
    avg_nll_conditional=float(avg_nll_conditional),
    # Deltas
    suffix_delta=float(suffix_delta),
    joint_delta=float(joint_delta),
    nll_delta=float(joint_delta),  # backward-compat alias for v00.01 dashboards
    # Reward decomposition
    reward_alpha_term=float(alpha * suffix_delta),
    reward_beta_term=float(beta * joint_delta),
    length_truncated=bool(length_truncated),
    length_penalty_applied=float(length_penalty_applied),
    # Token counts
    aux_tokens=len(aux_ids),
    aux_tokens_pre_trim=int(aux_tokens_pre_trim),
    prefix_tokens=len(prefix_ids),
    suffix_tokens=len(suffix_ids),
    cut_offset=int(cut),
    suffix_overlap_ratio=overlap_ratio,
)
```

`aux_tokens_pre_trim` should be captured BEFORE the budget-trim block at line 170–172
(i.e., `aux_tokens_pre_trim = len(aux_ids)` immediately after the policy-side tokenization
on line 164, before the overshoot check).

### 2. New configs `conf/latent-thought-v00.0{2..6}.yaml`

Strategy: full copy of `conf/latent-thought-v00.01.yaml` per config (matches the repo's existing
versioning pattern — every prior `vNN.NN.yaml` is a self-contained copy off `base`, not a chain
of inherits). For each new file change ONLY:

- `finetune.hub_model_revision: latent-thought-v00.0X`
- The `latent_thought:` block:
  ```yaml
  latent_thought:
      max_total_tokens: 16384
      reward_alpha: <α>
      reward_beta:  <β>
      length_penalty: <0.0 or 0.1>
  ```

Per-file values follow the table in the Context section.

Leave `conf/latent-thought-v00.01.yaml` untouched. The code defaults (`reward_alpha=0.0`,
`reward_beta=1.0`, `length_penalty=0.0`) reproduce the v00.01 reward exactly, so the existing
config remains valid without edits.

### 3. New launch scripts `train/RL/bash/latent-thought-v00.0{2..6}.sh`

Full copy of `train/RL/bash/latent-thought-v00.01.sh` per config (same convention). Only
two lines change in each:

- `export JOB_NAME="latent-thought-v00.0X"`
- `python -m pipelinerl.launch --config-name=latent-thought-v00.0X ...`

## Critical files

- `pipelinerl/domains/latent_thought/rollouts.py` — reward + penalty + metrics changes
- `conf/latent-thought-v00.0{2,3,4,5,6}.yaml` — five new configs
- `train/RL/bash/latent-thought-v00.0{2,3,4,5,6}.sh` — five new launch scripts
- `pipelinerl/domains/math/rollouts.py:50–56` — reference for length-penalty pattern (not modified)

## Verification

1. **Static check.** From the repo root:
   ```bash
   source /mnt/weka/home/wen.ye/workspace_m2/envs/uvs/prl/bin/activate
   python -c "from pipelinerl.domains.latent_thought import rollouts; print('ok')"
   ```

2. **Numerical regression.** Before changing v00.01 behavior, write a tiny scratch script that
   builds a fake `(prefix_ids, aux_ids, suffix_ids, baseline_lp, treatment_lp, conditional_lp)`
   and asserts that with `alpha=0, beta=1, length_penalty=0`, the new code returns the same
   `reward` value as the old formula. Confirms backward compatibility for v00.01 defaults.

3. **finish_reason path.** Inspect `LLMOutput` (likely under
   `tapeagents/llms/`) to confirm whether `finish_reason` is exposed. If not, document the
   fallback (`output_length_tokens >= max_tokens_cap`) in the rollout docstring.

4. **Smoke launch.** Submit v00.03 first (`alpha=1, beta=0, no penalty`) — the cleanest
   "new" reward. Watch the first ~50 steps in wandb for:
   - `suffix_delta` and `joint_delta` both populated and roughly the expected sign
   - evaluator queue not backed up (third logprob call)
   - `aux_tokens` distribution starting to drop (no more pure copy-prefix behavior)

5. **Sweep.** Once v00.03 is healthy, launch v00.02, v00.04, v00.05, v00.06.
   Compare in wandb under the existing `AI-CSI` project, tagged `latent_thought`.

## Project memory updates (post-launch)

- `project-memory/STATUS.md` — record the 5 new job IDs in **Waiting On** with the comparison
  goal and the exact wandb panel to inspect.
- `project-memory/experiments.md` — one row per config with: config path, command, commit,
  job ID, status, eventual key results.
- `project-memory/decisions.md` — log: hybrid `(α + β = 1)` reward, hard length penalty as
  config scalar, full-copy config strategy.
- `project-memory/hypotheses.md` — H1: suffix-only reward (α=1) eliminates the
  copy-prefix hack. H2: hard length penalty (0.1) shortens aux without hurting NLL improvement.
