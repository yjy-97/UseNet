
cd "$(dirname "$0")/.." || exit 1
echo "[INFO] Working directory: $(pwd)"

NUM_GPUS=${NUM_GPUS:-1}

GPU_UTIL_THRESHOLD=30    # GPU utilization % below this is considered idle
GPU_MEM_THRESHOLD=50     # GPU memory usage % below this is considered idle

select_idle_gpus() {
    local num_requested=$1

    if ! command -v nvidia-smi &> /dev/null; then
        echo "[GPU] nvidia-smi not found, defaulting to GPU 0" >&2
        echo "0"
        return
    fi

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
               --format=csv,noheader,nounits 2>/dev/null)

    if [ -z "$gpu_info" ]; then
        echo "[GPU] Failed to query GPU info, defaulting to GPU 0" >&2
        echo "0"
        return
    fi

    echo "[GPU] ========== GPU Status ==========" >&2

    local idle_gpus=()
    local all_gpus=()

    while IFS=',' read -r idx util mem_used mem_total; do
        idx=$(echo "$idx" | xargs)
        util=$(echo "$util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)

        if [ "$mem_total" -gt 0 ] 2>/dev/null; then
            mem_pct=$((mem_used * 100 / mem_total))
        else
            mem_pct=100
        fi

        local status="BUSY"
        if [ "$util" -lt "$GPU_UTIL_THRESHOLD" ] && [ "$mem_pct" -lt "$GPU_MEM_THRESHOLD" ]; then
            status="IDLE"
            idle_gpus+=("$idx")
        fi

        echo "[GPU]   GPU $idx: Util=${util}%, Mem=${mem_used}/${mem_total} MiB (${mem_pct}%) -> $status" >&2
        all_gpus+=("${util}:${idx}")
    done <<< "$gpu_info"

    echo "[GPU] ====================================" >&2

    local selected=()

    if [ ${#idle_gpus[@]} -ge "$num_requested" ]; then
        selected=("${idle_gpus[@]:0:$num_requested}")
        echo "[GPU] Found ${#idle_gpus[@]} idle GPU(s). Selected: ${selected[*]}" >&2
    elif [ ${#idle_gpus[@]} -gt 0 ]; then
        selected=("${idle_gpus[@]}")
        echo "[GPU] Warning: Only ${#idle_gpus[@]} idle GPU(s) found (requested $num_requested). Using: ${selected[*]}" >&2
    else
        echo "[GPU] Warning: No idle GPUs found! Selecting the least busy GPU(s)." >&2
        IFS=$'\n' sorted=($(printf '%s\n' "${all_gpus[@]}" | sort -t: -k1 -n)); unset IFS
        for entry in "${sorted[@]}"; do
            if [ ${#selected[@]} -ge "$num_requested" ]; then
                break
            fi
            local gpu_idx="${entry#*:}"
            selected+=("$gpu_idx")
        done
        echo "[GPU] Fallback selected: ${selected[*]}" >&2
    fi

    local result=""
    for i in "${!selected[@]}"; do
        if [ "$i" -gt 0 ]; then
            result+=","
        fi
        result+="${selected[$i]}"
    done
    echo "$result"
}

SELECTED_GPUS=$(select_idle_gpus $NUM_GPUS)
export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS

echo ""
echo "=============================================="
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "=============================================="
echo ""

IFS=',' read -ra GPU_ARRAY <<< "$SELECTED_GPUS"
MULTI_GPU_FLAG=""
if [ ${#GPU_ARRAY[@]} -gt 1 ]; then
    MULTI_GPU_FLAG="--use_multi_gpu --devices $SELECTED_GPUS"
    echo "[GPU] Multi-GPU mode enabled with devices: $SELECTED_GPUS"
fi



python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/sharedata/EEG_Data_GZ/APAVA/APAVA/ \
  --model_id APAVA-Indep \
  --model iTransformer \
  --data APAVA \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  $MULTI_GPU_FLAG

python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/sharedata/shenkaiye/TDBrain/ \
  --model_id TDBRAIN-Indep \
  --model iTransformer \
  --data TDBRAIN \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  $MULTI_GPU_FLAG

python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/sharedata/EEG_Data_GZ/ADFD/ADFD2class/ \
  --model_id ADFTD-Indep \
  --model iTransformer \
  --data ADFTD \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  $MULTI_GPU_FLAG

python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/sharedata/EEG_Data_GZ/ADSZ/ADSZ/ \
  --model_id ADSZ-Indep \
  --model iTransformer \
  --data ADSZ \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  $MULTI_GPU_FLAG

python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/sharedata/EEG_Data_GZ/MCICN/MCICN/ \
  --model_id MCICN-Indep \
  --model iTransformer \
  --data MCICN \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  $MULTI_GPU_FLAG
