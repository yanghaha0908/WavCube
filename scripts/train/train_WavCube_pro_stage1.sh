MASTER_ADDR="127.0.0.1"
MASTER_PORT=12345
NUM_GPUS=${NUM_GPUS:-8}
EXP_NAME="WavCube-pro-stage1"

LOG_DIR="logs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

echo "🚀 Single-node training | GPUs=${NUM_GPUS} | Exp=${EXP_NAME}"

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
    2>&1 | tee "${LOG_DIR}/train.log"
