#!/usr/bin/env bash
# SD3.5-Medium FlowGRPO OCR LoRA on Intel GPU, with the GPU recipe's 3-GPU layout
# (2 for actor + rollout, 1 for the reward model). Setup: docs/start/install_xpu.md.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Sleep mode is not supported on Intel GPU yet, so the rollout keeps the SD3.5
# weights on the GPU while the actor trains. The actor offloads its parameters
# and uses smaller micro batches, and the rollout batches fewer requests, to fit
# next to them.
# vLLM profiles the OCR reward model's vision encoder with its largest image;
# capping max_pixels (the rollout images are 384x384) keeps that within 24 GB.
bash "$SCRIPT_DIR/run_sd35_medium_ocr_lora_v1.sh" \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=8 \
    +reward.models.ocr.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=401408 \
    'trainer.logger=["console","tensorboard"]' \
    trainer.experiment_name=sd35_medium_ocr_lora_v1_xpu \
    "$@"
