set -e
cd "$(dirname "$0")/.."

DATASET=${DATASET:-APAVA}
ROOT_PATH=${ROOT_PATH:-/path/to/APAVA/}
SAMPLING_RATE=${SAMPLING_RATE:-128}
CHANNEL_CONFIG=${CHANNEL_CONFIG:-configs_channel_layout_example.json}

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "$ROOT_PATH" \
  --model_id "${DATASET}-VEUS" \
  --model UseNet \
  --data "$DATASET" \
  --sampling_rate "$SAMPLING_RATE" \
  --channel_config "$CHANNEL_CONFIG" \
  --virtual_channels 19 \
  --frequency_bands "1-4,4-8,8-13,13-30,30-45" \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 8 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --channel_drop_min 0.10 \
  --channel_drop_max 0.50 \
  --regional_drop_probability 0.50 \
  --lambda_kd 0.50 \
  --lambda_mapping 0.10 \
  --lambda_band 0.10 \
  --des VEUS_Exp \
  --itr 1
