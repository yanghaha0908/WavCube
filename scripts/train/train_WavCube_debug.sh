EXP_NAME="WavCube-pro-stage1"
LOG_DIR="logs/${EXP_NAME}_debug"
mkdir -p "$LOG_DIR"

python train.py \
    -c "configs/${EXP_NAME}.yaml" \
    --trainer.num_nodes=1 \
    --trainer.devices=1 \
    2>&1 | tee "${LOG_DIR}/train_debug.log"

# -m debugpy --wait-for-client --listen 5679 