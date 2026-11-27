#!/usr/bin/env bash
# SD3.5-Medium FlowGRPO OCR LoRA (v1 sync) on 3x Intel Arc Pro B60 (24 GB).
#
# Runs run_sd35_medium_ocr_lora_v1.sh unchanged plus two memory overrides:
#   - rollout max_num_seqs 256 -> 8: VAE decode runs out of memory next to the
#     resident weights at 256.
#   - reward model max_pixels 401408: vLLM profiles Qwen2.5-VL's vision encoder
#     with its largest image, which does not fit on 24 GB.
#
# Layout: 2 GPUs actor + rollout, 1 GPU reward (same as the base script).
# Prepare data and models as in the FlowGRPO quickstart, then:
#   bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_xpu_b60.sh
# Extra arguments are passed through to the base script.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "${SCRIPT_DIR}/run_sd35_medium_ocr_lora_v1.sh" \
    actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=8 \
    +reward.models.ocr.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=401408 \
    "$@"
