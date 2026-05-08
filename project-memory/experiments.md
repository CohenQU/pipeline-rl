# Experiments — pipeline-rl

<!-- Format for each experiment: -->
<!-- ## EXP-NNN: Short description -->
<!-- - **Date**: YYYY-MM-DD -->
<!-- - **Status**: queued | running | done | failed | abandoned | invalid -->
<!-- - **Hypothesis**: What hypothesis does this test? (link to hypotheses.md) -->
<!-- - **Goal**: What are we testing? -->
<!-- - **Config**: Key hyperparameters or config file path -->
<!-- - **Seeds**: Random seeds used -->
<!-- - **Command**: The launch command -->
<!-- - **Job ID**: Slurm/cluster job ID -->
<!-- - **Code snapshot**: Git commit hash or branch name -->
<!-- - **Output path**: Where results are saved -->
<!-- - **Checkpoint path**: Where model checkpoints are saved -->
<!-- - **Launch time**: When the job was submitted -->
<!-- - **Results**: Key metrics or findings -->
<!-- - **Valid**: Yes/No — if No, explain why results are invalid -->
<!-- - **Notes**: Anything worth remembering -->

## EXP-001: latent-thought-v00.01 baseline (joint reward, no length penalty)
- **Date**: 2026-04-30
- **Status**: done (showed reward hacking)
- **Hypothesis**: HYP-001 (this run produced the symptom that motivated the sweep)
- **Goal**: First end-to-end run of latent_thought RL on Qwen3-4B-Instruct-2507 with frozen Qwen3-4B-Base evaluator.
- **Config**: `conf/latent-thought-v00.01.yaml` (α=0, β=1, length_penalty=0; reward = `joint_delta` only)
- **Command**: `sbatch train/RL/bash/latent-thought-v00.01.sh`
- **Code snapshot**: commit `6fe0f90` (Always emit a training_text in latent_thought rollouts)
- **Output path**: `/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v00.01/`
- **Hub revision**: `lm-provers/QED-Nano-v1.6-dev@latent-thought-v00.01`
- **Results**: Reward hack — policy generates very long aux (often saturating the 1024 max_tokens cap) that copies large chunks of PREFIX. Inflates `joint_delta` via denominator dilution without genuinely improving suffix predictability.
- **Valid**: Yes (the hack is the finding)
- **Notes**: Drove the v00.02–v00.06 redesign (DEC-001, DEC-002).

## EXP-002: latent-thought-v00.02 (joint reward + length penalty 0.1)
- **Date**: 2026-05-02 (submitted)
- **Status**: submitted, pending — Job ID **1595300** (QOSGrpNodeLimit)
- **Hypothesis**: HYP-002 (isolates the length-penalty effect on the v00.01 reward)
- **Goal**: Test whether the hard length penalty alone (without changing the reward) is enough to suppress the copy-prefix hack.
- **Config**: `conf/latent-thought-v00.02.yaml` (α=0, β=1, length_penalty=0.1)
- **Command**: `sbatch train/RL/bash/latent-thought-v00.02.sh`
- **Job ID**: 1596542 (resubmitted after DEC-004 fix; 1595300 cancelled along with the v00.03 deadlock)
- **Code snapshot**: branch `aicsi` at commit 975c3dd + uncommitted preprocess fix
- **Output path**: `/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v00.02/`
- **Logs**: `/mnt/weka/home/wen.ye/workspace_m2/tmp/log/slurm/sft-1596542.out`

## EXP-003: latent-thought-v00.03 (suffix-only reward, no length penalty) — primary smoke run
- **Date**: 2026-05-02 (submitted)
- **Status**: submitted, pending — Job ID **1595298** (QOSGrpNodeLimit)
- **Hypothesis**: HYP-001 (cleanest test of "is the dilution the hack?")
- **Goal**: Replace joint_delta with suffix_delta entirely; expect copy-prefix behavior to disappear because there is no longer a way to inflate the reward via long aux.
- **Config**: `conf/latent-thought-v00.03.yaml` (α=1, β=0, length_penalty=0)
- **Command**: `sbatch train/RL/bash/latent-thought-v00.03.sh`
- **Job ID**: 1596540 (resubmitted after DEC-004 fix; 1595298 cancelled — deadlocked at preprocess gate, see history.md 2026-05-02)
- **Code snapshot**: branch `aicsi` at commit 975c3dd + uncommitted preprocess fix
- **Output path**: `/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v00.03/`
- **Logs**: `/mnt/weka/home/wen.ye/workspace_m2/tmp/log/slurm/sft-1596540.out`
- **Notes**: Watch first ~50 steps in wandb (project AI-CSI, tag `latent_thought`): `suffix_delta`, `joint_delta`, `aux_tokens`, evaluator queue (3 prompt_logprobs calls per rollout vs 2 in v00.01). v00.03 v1 (1595298) wasted 4h 24m in deadlock before being killed — see DEC-004 / errors_and_fixes.md.

## EXP-004: latent-thought-v00.04 (suffix-only reward + length penalty 0.1)
- **Date**: 2026-05-02 (queued)
- **Status**: queued
- **Hypothesis**: HYP-002 (does penalty further shorten aux on top of suffix-only reward?)
- **Config**: `conf/latent-thought-v00.04.yaml` (α=1, β=0, length_penalty=0.1)
- **Command**: `sbatch train/RL/bash/latent-thought-v00.04.sh`

## EXP-005: latent-thought-v00.05 (50/50 hybrid, no length penalty)
- **Date**: 2026-05-02 (submitted)
- **Status**: submitted, pending — Job ID **1595299** (QOSGrpNodeLimit)
- **Hypothesis**: HYP-003 (does the joint term reduce gradient noise without reintroducing the hack?)
- **Config**: `conf/latent-thought-v00.05.yaml` (α=0.5, β=0.5, length_penalty=0)
- **Command**: `sbatch train/RL/bash/latent-thought-v00.05.sh`
- **Job ID**: 1596541 (resubmitted after DEC-004 fix; 1595299 cancelled along with the v00.03 deadlock)
- **Code snapshot**: branch `aicsi` at commit 975c3dd + uncommitted preprocess fix
- **Output path**: `/mnt/weka/home/wen.ye/workspace_m2/tmp/models/latent-thought-v00.05/`
- **Logs**: `/mnt/weka/home/wen.ye/workspace_m2/tmp/log/slurm/sft-1596541.out`

## EXP-006: latent-thought-v00.06 (50/50 hybrid + length penalty 0.1)
- **Date**: 2026-05-02 (queued)
- **Status**: queued
- **Hypothesis**: HYP-002 + HYP-003
- **Config**: `conf/latent-thought-v00.06.yaml` (α=0.5, β=0.5, length_penalty=0.1)
- **Command**: `sbatch train/RL/bash/latent-thought-v00.06.sh`

## EXP-007: latent-thought-v00.07 (suffix-only + small fluency regularizer)
- **Date**: 2026-05-08 (queued)
- **Status**: ready to submit
- **Hypothesis**: HYP-004 (small γ improves stability without re-introducing copy-prefix)
- **Goal**: Test whether γ=0.1 on aux_delta gently improves training over pure suffix-only (v00.03).
- **Config**: `conf/latent-thought-v00.07.yaml` (α=0.9, β=0, γ=0.1, length_penalty=0)
- **Command**: `sbatch train/RL/bash/latent-thought-v00.07.sh`
- **Code snapshot**: branch `aicsi`, will record commit at submission

## EXP-008: latent-thought-v00.08 (suffix-only + moderate fluency regularizer)
- **Date**: 2026-05-08 (queued)
- **Status**: ready to submit
- **Hypothesis**: HYP-004
- **Goal**: γ=0.3 — middle of the sweep. Should still favor suffix prediction but with stronger fluency pressure.
- **Config**: `conf/latent-thought-v00.08.yaml` (α=0.7, β=0, γ=0.3, length_penalty=0)
- **Command**: `sbatch train/RL/bash/latent-thought-v00.08.sh`

## EXP-009: latent-thought-v00.09 (50/50 suffix and aux fluency)
- **Date**: 2026-05-08 (queued)
- **Status**: ready to submit
- **Hypothesis**: HYP-004 (upper bound — copy-prefix pathology should start to emerge here if it's going to)
- **Goal**: γ=0.5 — equal weight on suffix delta and aux fluency. Watch for `suffix_overlap_ratio` increasing (aux drifting toward prefix copy).
- **Config**: `conf/latent-thought-v00.09.yaml` (α=0.5, β=0, γ=0.5, length_penalty=0)
- **Command**: `sbatch train/RL/bash/latent-thought-v00.09.sh`

