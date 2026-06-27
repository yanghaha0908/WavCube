MASTER_ADDR="127.0.0.1"
MASTER_PORT=12345
NUM_GPUS=${NUM_GPUS:-8}
EXP_NAME="WavCube-pro-stage2"

# Stage 2 需要从 Stage 1 的 checkpoint 恢复
RESUME_CKPT="../vocos/logs/wavlmvae-mimo-librispeech-stage1_kl1e-4_ae_300mdeco_6k/first/version_2/checkpoints/vocos_checkpoint_epoch=41_step=138000_val_loss=6.2627.ckpt"


LOG_DIR="logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

echo "🚀 Single-node training | GPUs=${NUM_GPUS} | Exp=${EXP_NAME}"
echo "📦 Resume from: ${RESUME_CKPT}"

torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    -c "configs/${EXP_NAME}.yaml" \
    --trainer.num_nodes=1 \
    --trainer.devices=${NUM_GPUS} \
    --trainer.strategy=ddp \
    --model.feature_extractor.init_args.stage1_ckpt_path=${RESUME_CKPT} \
    2>&1 | tee "${LOG_DIR}/train.log"
