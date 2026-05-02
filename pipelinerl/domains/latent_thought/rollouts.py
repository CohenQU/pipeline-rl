"""
Rollout function for the latent_thought domain.

Given (prefix, suffix) sliced from a raw text row, the policy generates an
auxiliary string `aux`. A frozen evaluator vLLM scores three quantities:
  - baseline:    `suffix | prefix`            (no aux conditioning)
  - treatment:   `(aux ⊕ suffix) | prefix`    (joint scoring, length-diluted)
  - conditional: `suffix | prefix, aux`       (suffix-only, aux as context)

Reward is a hybrid weighted sum of two deltas, optionally docked by a hard
length penalty when the policy hit its `max_tokens` cap:

  avg_NLL_baseline    = -sum_logprob(suffix | prefix)          / |suffix|
  avg_NLL_treatment   = -sum_logprob(aux⊕suffix | prefix)      / (|aux| + |suffix|)
  avg_NLL_conditional = -sum_logprob(suffix | prefix, aux)     / |suffix|

  suffix_delta = avg_NLL_baseline - avg_NLL_conditional
  joint_delta  = avg_NLL_baseline - avg_NLL_treatment
  reward       = α · suffix_delta + β · joint_delta - length_penalty_applied

with the constraint `α + β = 1` (asserted at runtime).

Why hybrid: v00.01 used reward = joint_delta only. That formula admits a
copy-prefix hack — the model emits long aux that is trivially predictable
under the evaluator, which inflates the (|aux| + |suffix|) denominator with
low-NLL tokens and lowers avg_NLL_treatment without making the suffix any
easier to predict. suffix_delta divides both sides by |suffix|, so it is
not dilutable and directly measures "did aux help predict the suffix?".

Length penalty: when policy generation finishes due to `max_tokens` (vs.
natural EOS), `length_penalty` is subtracted from the reward. Detection
falls back to comparing `llm_call.output_length_tokens` to
`cfg.llm.parameters.max_tokens` (LLMOutput does not carry finish_reason in
this stack). Default `length_penalty=0.0` (off); typical comparison value
is `0.1`.

Backward compatibility: defaults `reward_alpha=0.0, reward_beta=1.0,
length_penalty=0.0` reproduce the v00.01 reward exactly.

`suffix_overlap_ratio` continues to be logged for monitoring; it is NOT
subtracted from the reward.
"""

import asyncio
import difflib
import logging
import random
import time

import aiohttp
from omegaconf import DictConfig
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import RolloutResult, BaseMetrics

logger = logging.getLogger(__name__)


class Metrics(BaseMetrics):
    avg_nll_baseline: float = 0.0
    avg_nll_treatment: float = 0.0
    avg_nll_conditional: float = 0.0
    suffix_delta: float = 0.0
    joint_delta: float = 0.0
    nll_delta: float = 0.0
    reward_alpha_term: float = 0.0
    reward_beta_term: float = 0.0
    length_truncated: bool = False
    length_penalty_applied: float = 0.0
    aux_tokens: int = 0
    aux_tokens_pre_trim: int = 0
    prefix_tokens: int = 0
    suffix_tokens: int = 0
    cut_offset: int = 0
    suffix_overlap_ratio: float = 0.0


def remove_reasoning(completion: str, reasoning_delimiters: list[str] | None = None) -> str:
    """Strip reasoning prefix (e.g. </think> content) from a completion."""
    if not reasoning_delimiters:
        return completion
    for delim in reasoning_delimiters:
        if delim in completion:
            return completion.split(delim)[-1].strip()
    return ""


def _sum_logprobs(result_dict: dict) -> float:
    """Extract sum of logprobs from a get_batch_logprobs_token_ids result entry.

    Response shape: {"content": [{"logprob": float, "token_id": str, ...}, ...]}
    """
    return sum(item["logprob"] for item in result_dict["content"])


def _truncate_to_fit(
    prefix_ids: list[int],
    suffix_ids: list[int],
    max_total: int,
) -> tuple[list[int], list[int]]:
    """Trim the longer of prefix/suffix until prefix+suffix fits within max_total.

    Truncates prefix from the LEFT (drop oldest tokens) and suffix from the RIGHT
    (drop trailing tokens). Returns (prefix_ids, suffix_ids).
    """
    while len(prefix_ids) + len(suffix_ids) > max_total:
        if len(prefix_ids) >= len(suffix_ids) and len(prefix_ids) > 0:
            prefix_ids = prefix_ids[1:]
        elif len(suffix_ids) > 0:
            suffix_ids = suffix_ids[:-1]
        else:
            break
    return prefix_ids, suffix_ids


async def generate_latent_thought_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
    *,
    evaluator_llm: TrainableLLM,
) -> RolloutResult:
    """Latent-thought rollout. See module docstring for the reward formula."""
    time_start = time.time()

    lt_cfg = cfg.get("latent_thought", {}) or {}
    max_total_tokens: int = int(lt_cfg.get("max_total_tokens", 16384))
    # Filtering happens once at dataset construction time (loader's
    # min_side_chars / max_total_chars). Once a problem reaches this rollout we
    # always emit one training_text — skipping here would desync the preprocessor's
    # group-size invariant (every rollout in a group of `attempts` must contribute
    # a sample with a rollout_index).

    # ----- 1. Pick a cut online ---------------------------------------------
    cut_offsets = list(problem.get("cut_offsets") or [])
    text = problem["text"]
    cut = random.choice(cut_offsets) if cut_offsets else max(1, len(text) // 2)
    prefix_text = text[:cut]
    suffix_text = text[cut:]

    # ----- 2. Token-level enforcement ---------------------------------------
    evaluator_llm.load_tokenizer()
    eval_tokenizer = evaluator_llm.tokenizer

    prefix_ids = eval_tokenizer(prefix_text, add_special_tokens=True).input_ids
    suffix_ids = eval_tokenizer(suffix_text, add_special_tokens=False).input_ids

    if len(prefix_ids) + len(suffix_ids) > max_total_tokens:
        prefix_ids, suffix_ids = _truncate_to_fit(prefix_ids, suffix_ids, max_total_tokens)

    # Defensive: if either side is empty after tokenization (extremely short
    # text or aggressive truncation), pad with a single special token so the
    # evaluator call is well-formed. Reward will be near zero on these edge
    # cases, but we still emit a training_text.
    if len(prefix_ids) == 0:
        prefix_ids = [eval_tokenizer.bos_token_id or eval_tokenizer.eos_token_id]
    if len(suffix_ids) == 0:
        suffix_ids = [eval_tokenizer.eos_token_id]

    # Re-derive text from token ids in case we truncated; the policy sees the
    # same text the evaluator will score over.
    prefix_text_view = eval_tokenizer.decode(prefix_ids, skip_special_tokens=False)
    suffix_text_view = eval_tokenizer.decode(suffix_ids, skip_special_tokens=False)

    # ----- 3. Build policy prompt and generate aux --------------------------
    messages = []
    if cfg.actor.system_prompt is not None:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({
        "role": "user",
        "content": cfg.actor.task_template.format(
            prefix=prefix_text_view,
            suffix=suffix_text_view,
        ),
    })
    prompt = Prompt(messages=messages)

    llm_call = await llm_async_generate(llm, prompt, session)
    assert llm_call.output.content is not None
    aux_raw = llm_call.output.content

    # Strip reasoning if the policy is a thinking model.
    reasoning_delimiters = None
    llm_cfg = cfg.get("llm", None)
    if llm_cfg is not None:
        reasoning_delimiters = llm_cfg.get("reasoning_delimiters", None)
    aux_text = remove_reasoning(aux_raw, reasoning_delimiters=reasoning_delimiters)
    if not aux_text.strip():
        aux_text = aux_raw  # fall back to raw output for non-thinking models

    aux_ids = eval_tokenizer(aux_text, add_special_tokens=False).input_ids
    aux_tokens_pre_trim = len(aux_ids)

    # ----- 4. Reward via evaluator ------------------------------------------
    # Cap (prefix + aux + suffix) at the same max_total_tokens; trim aux first
    # if the joint blows past the budget. (Trim aux from the right so that the
    # suffix portion of the joint stays intact.)
    overshoot = (len(prefix_ids) + len(aux_ids) + len(suffix_ids)) - max_total_tokens
    if overshoot > 0:
        aux_ids = aux_ids[: max(0, len(aux_ids) - overshoot)]

    # Three prompt_logprobs calls per rollout. Higher evaluator throughput
    # pressure than v00.01 — monitor evaluator queue length on first run.
    baseline_result, treatment_result, conditional_result = await asyncio.gather(
        asyncio.to_thread(
            evaluator_llm.get_batch_logprobs_token_ids,
            [prefix_ids],
            [suffix_ids],
        ),
        asyncio.to_thread(
            evaluator_llm.get_batch_logprobs_token_ids,
            [prefix_ids],
            [aux_ids + suffix_ids],
        ),
        asyncio.to_thread(
            evaluator_llm.get_batch_logprobs_token_ids,
            [prefix_ids + aux_ids],
            [suffix_ids],
        ),
    )
    sum_lp_baseline = _sum_logprobs(baseline_result[0])
    sum_lp_treatment = _sum_logprobs(treatment_result[0])
    sum_lp_conditional = _sum_logprobs(conditional_result[0])

    n_suffix = max(1, len(suffix_ids))
    n_joint = max(1, len(aux_ids) + len(suffix_ids))
    avg_nll_baseline = -sum_lp_baseline / n_suffix
    avg_nll_treatment = -sum_lp_treatment / n_joint
    avg_nll_conditional = -sum_lp_conditional / n_suffix

    suffix_delta = avg_nll_baseline - avg_nll_conditional
    joint_delta = avg_nll_baseline - avg_nll_treatment

    alpha = float(lt_cfg.get("reward_alpha", 0.0))
    beta = float(lt_cfg.get("reward_beta", 1.0))
    assert abs(alpha + beta - 1.0) < 1e-6, (
        f"latent_thought.reward_alpha + reward_beta must equal 1.0, got {alpha + beta}"
    )
    reward = alpha * suffix_delta + beta * joint_delta

    # Hard length penalty when the policy stopped because it hit max_tokens.
    # LLMOutput in this stack does not carry finish_reason, so detect via
    # output_length_tokens >= configured cap (math domain uses an analogous
    # token-count check at pipelinerl/domains/math/rollouts.py:50–56).
    length_penalty_value = float(lt_cfg.get("length_penalty", 0.0))
    max_tokens_cap = int(cfg.llm.parameters.max_tokens)
    length_truncated = bool(llm_call.output_length_tokens >= max_tokens_cap)
    length_penalty_applied = length_penalty_value if length_truncated else 0.0
    reward = reward - length_penalty_applied

    discount_factor = float(cfg.actor.get("discount_factor", 1.0))
    if discount_factor != 1.0:
        reward = reward * (discount_factor ** llm_call.output_length_tokens)

    # ----- 5. Suffix-overlap monitoring (NOT used in reward in v0) ----------
    overlap_ratio = 0.0
    if aux_text and suffix_text_view:
        overlap_ratio = float(
            difflib.SequenceMatcher(None, aux_text, suffix_text_view, autojunk=False).ratio()
        )

    # ----- 6. Build training text and metrics -------------------------------
    trace = make_training_text(llm, llm_call)
    trace.reward = float(reward)

    metrics = Metrics(
        reward=float(reward),
        success=reward > 0.0,
        no_error=True,
        no_answer=not bool(aux_text.strip()),
        avg_nll_baseline=float(avg_nll_baseline),
        avg_nll_treatment=float(avg_nll_treatment),
        avg_nll_conditional=float(avg_nll_conditional),
        suffix_delta=float(suffix_delta),
        joint_delta=float(joint_delta),
        nll_delta=float(joint_delta),  # backward-compat alias for v00.01 dashboards
        reward_alpha_term=float(alpha * suffix_delta),
        reward_beta_term=float(beta * joint_delta),
        length_truncated=length_truncated,
        length_penalty_applied=float(length_penalty_applied),
        aux_tokens=len(aux_ids),
        aux_tokens_pre_trim=int(aux_tokens_pre_trim),
        prefix_tokens=len(prefix_ids),
        suffix_tokens=len(suffix_ids),
        cut_offset=int(cut),
        suffix_overlap_ratio=overlap_ratio,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=time.time() - time_start,
        dataset_name=problem.get("dataset"),
    )
