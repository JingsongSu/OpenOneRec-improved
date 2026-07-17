#!/usr/bin/env bash
set -euo pipefail
set -x

PRETRAIN_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-latent/pretrain

# ============================================================
# 模型路径
# ============================================================

# Stage2 转换后的原始 HuggingFace 模型。
# 该目录用于执行 tools/prepare_pause_model.py，
# 不直接作为 Pause SFT 的训练输入。
STAGE2_OUTPUT_DIR=${PRETRAIN_DIR}/model_output/stg2
STAGE2_MODEL_DIR=${STAGE2_OUTPUT_DIR}/step11500/global_step11500/converted

# 已经通过 tools/prepare_pause_model.py 加入
# <|latent_pause|> token 后的 HuggingFace 模型。
MODEL_DIR=${PRETRAIN_DIR}/model_output/stg2_pause_ready

# ============================================================
# SFT 输出与数据配置
# ============================================================

# Pause Latent SFT 输出目录。
OUTPUT_DIR=${PRETRAIN_DIR}/model_output/sft_pause_k5-0.2

# 已经通过 tools/make_pause_dataset_config.py 生成的数据配置。
DATASET_CONFIG=${PRETRAIN_DIR}/examples/dataset_config/sft_pause.json

# ============================================================
# Pause Latent 配置
# ============================================================

# 第一轮实验建议设置为 0.0：
# 只使用正常的 target-token CE。
# Itemic token 的 loss 仍然会通过 attention 回传到全部 pause hidden states。
PAUSE_PLAN_AUX_WEIGHT=0.2

# 当启用 pause planning auxiliary loss 时，
# 每张卡、每个 step 最多计算多少组 pause-to-Itemic 辅助预测。
PAUSE_PLAN_MAX_PAIRS=64

# 辅助 planning loss 的 warmup step。
PAUSE_PLAN_WARMUP_STEPS=200

# OpenOneRec 当前 Itemic token ID 范围。
PAUSE_ITEMIC_START=151669
PAUSE_ITEMIC_END=169742

cd "${PRETRAIN_DIR}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /tmp/_wids_cache

# ============================================================
# 启动前检查
# ============================================================

if [[ ! -d "${STAGE2_MODEL_DIR}" ]]; then
    echo "ERROR: Stage2 converted model directory does not exist:"
    echo "${STAGE2_MODEL_DIR}"
    exit 1
fi

if [[ ! -f "${STAGE2_MODEL_DIR}/config.json" ]]; then
    echo "ERROR: Stage2 config.json not found:"
    echo "${STAGE2_MODEL_DIR}/config.json"
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Prepared pause model directory does not exist:"
    echo "${MODEL_DIR}"
    echo ""
    echo "Please run tools/prepare_pause_model.py first."
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: Prepared pause model config.json not found:"
    echo "${MODEL_DIR}/config.json"
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/pause_prefix_config.json" ]]; then
    echo "ERROR: pause_prefix_config.json not found:"
    echo "${MODEL_DIR}/pause_prefix_config.json"
    echo ""
    echo "The model may not have been prepared with:"
    echo "tools/prepare_pause_model.py"
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/tokenizer_config.json" ]]; then
    echo "ERROR: tokenizer_config.json not found:"
    echo "${MODEL_DIR}/tokenizer_config.json"
    exit 1
fi

if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "ERROR: Pause SFT dataset config not found:"
    echo "${DATASET_CONFIG}"
    echo ""
    echo "Please run tools/make_pause_dataset_config.py first."
    exit 1
fi

if [[ ! -f "${PRETRAIN_DIR}/recipes/train_qwen3_pause.py" ]]; then
    echo "ERROR: Pause training recipe not found:"
    echo "${PRETRAIN_DIR}/recipes/train_qwen3_pause.py"
    exit 1
fi

if [[ ! -f "${PRETRAIN_DIR}/torchrun_ompi_wrapper.py" ]]; then
    echo "ERROR: torchrun_ompi_wrapper.py not found:"
    echo "${PRETRAIN_DIR}/torchrun_ompi_wrapper.py"
    exit 1
fi

# ============================================================
# GPU 与 NCCL 设置
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_DEBUG=WARN

# 单机训练不需要跨节点 IB。
export NCCL_IB_DISABLE=1

# 保留 GPU P2P 和共享内存通信。
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# NCCL 异常检测。
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ============================================================
# torchrun rendezvous 设置
# ============================================================

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8501

# ============================================================
# 其他运行环境
# ============================================================

export PYTHONPATH=${PRETRAIN_DIR}:${PYTHONPATH:-}
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# 注意：
# 不要 source set_env.sh
# 不要使用 mpirun
# 不要使用 scripts/numa_runner.sh

# ============================================================
# 日志路径
# ============================================================

STDOUT_LOG=${OUTPUT_DIR}/stdout.log
STDERR_LOG=${OUTPUT_DIR}/stderr.log

SCRIPT_FILE=$(readlink -f "$0")

# ============================================================
# 记录任务信息
# ============================================================

{
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "script: ${SCRIPT_FILE}"
    echo "stage: Pause Latent SFT"
    echo "stage2_model_dir: ${STAGE2_MODEL_DIR}"
    echo "prepared_pause_model_dir: ${MODEL_DIR}"
    echo "output_dir: ${OUTPUT_DIR}"
    echo "dataset_config: ${DATASET_CONFIG}"
    echo "pause_prefix_config: ${MODEL_DIR}/pause_prefix_config.json"
    echo "pause_plan_aux_weight: ${PAUSE_PLAN_AUX_WEIGHT}"
    echo "pause_plan_max_pairs: ${PAUSE_PLAN_MAX_PAIRS}"
    echo "pause_plan_warmup_steps: ${PAUSE_PLAN_WARMUP_STEPS}"
    echo "pause_itemic_start: ${PAUSE_ITEMIC_START}"
    echo "pause_itemic_end: ${PAUSE_ITEMIC_END}"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
    echo "master_addr: ${MASTER_ADDR}"
    echo "master_port: ${MASTER_PORT}"
    echo "========================="
} >> "${OUTPUT_DIR}/task_info.log"

echo "============================================================"
echo "OpenOneRec Pause Latent SFT"
echo "============================================================"
echo "PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "STAGE2_MODEL_DIR=${STAGE2_MODEL_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATASET_CONFIG=${DATASET_CONFIG}"
echo "PAUSE_PLAN_AUX_WEIGHT=${PAUSE_PLAN_AUX_WEIGHT}"
echo "PAUSE_PLAN_MAX_PAIRS=${PAUSE_PLAN_MAX_PAIRS}"
echo "PAUSE_PLAN_WARMUP_STEPS=${PAUSE_PLAN_WARMUP_STEPS}"
echo "PAUSE_ITEMIC_START=${PAUSE_ITEMIC_START}"
echo "PAUSE_ITEMIC_END=${PAUSE_ITEMIC_END}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "STDOUT_LOG=${STDOUT_LOG}"
echo "STDERR_LOG=${STDERR_LOG}"
echo "============================================================"

# ============================================================
# 启动 Pause Latent SFT
# ============================================================

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --max_restarts=0 \
    torchrun_ompi_wrapper.py recipes/train_qwen3_pause.py \
        --model_dir "${MODEL_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --dataset_config "${DATASET_CONFIG}" \
        --use_tie_weights \
        --model_class Qwen3ForCausalLM \
        --monitor_datasource_loss \
        --monitor_datasource_cnt \
        --max_length 32768 \
        --learning_rate 2e-4 \
        --min_lr 1e-4 \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 500 \
        --num_training_steps 5000 \
        --save_checkpoint_per_step 500 \
        --minibatch_size 16384 \
        --logging_per_step 50 \
        --use_fp32_weight \
        --seed 19260817 \
        --enable_profiler \
        --enable_gradient_checkpointing \
        --use_chunked_loss_computer \
        --pause_plan_aux_weight "${PAUSE_PLAN_AUX_WEIGHT}" \
        --pause_plan_max_pairs "${PAUSE_PLAN_MAX_PAIRS}" \
        --pause_plan_warmup_steps "${PAUSE_PLAN_WARMUP_STEPS}" \
        --pause_itemic_start "${PAUSE_ITEMIC_START}" \
        --pause_itemic_end "${PAUSE_ITEMIC_END}" \
        > "${STDOUT_LOG}" \
        2> "${STDERR_LOG}"
