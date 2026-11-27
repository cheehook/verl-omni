#!/usr/bin/env bash
# SD3.5-Medium FlowGRPO OCR LoRA (v1 separate_async) on 5x Intel Arc Pro B60 (24 GB).
#
# Runs run_sd35_medium_ocr_lora_v1_separate_async.sh unchanged plus these overrides:
#   - CKPT_BACKEND=xccl: selects the XCCL checkpoint engine from
#     verl-hardware-plugin and skips the base script's CUDA (cupy) pre-check.
#   - rollout max_num_seqs 256 -> 8: VAE decode memory next to resident weights.
#   - reward model max_pixels 401408: caps vLLM's vision-encoder profiling for
#     Qwen2.5-VL on 24 GB.
#   - checkpoint engine bucket 2048 -> 512 MB: the default receive buffer does not
#     fit next to the SD3.5 pipeline on a standalone rollout GPU.
#   - actor param + optimizer offload, actor micro batch 1, log_prob and ref
#     micro batch 2: room for the actor on GPUs that also hold a colocated
#     rollout replica (micro batch 2 ran out of memory at step 94).
#
# Layout: 2 GPUs actor + colocated rollout, 2 GPUs standalone rollout,
# 1 GPU reward (same as the base script).
# Prepare data and models as in the FlowGRPO quickstart, then:
#   bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async_xpu_b60.sh
# Extra arguments are passed through to the base script.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export CKPT_BACKEND=${CKPT_BACKEND:-xccl}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}

exec bash "${SCRIPT_DIR}/run_sd35_medium_ocr_lora_v1_separate_async.sh" \
    +reward.reward_model.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=401408 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    "$@"
