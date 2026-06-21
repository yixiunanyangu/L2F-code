export DEBUG_MODE="true"


RUN_NAME="end_to_end_base_target+1"
export LOG_PATH="/workspace/VLM-R1/output/debug_log_$RUN_NAME.txt"

# 增加NCCL配置
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_TIMEOUT=1800  # 增加超时到30分钟
# export NCCL_ASYNC_ERROR_HANDLING=0  # 禁用异步错误处理
# export NCCL_MAX_NCHANNELS=4  # 限制通道数
# export CUDA_LAUNCH_BLOCKING=1  # 同步执行便于调试
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    /workspace/VLM-R1/src/open-r1-multimodal/src/open_r1/end_to_end_change_prompt.py \
    --deepspeed /workspace/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir /workspace/VLM-R1/output/$RUN_NAME \
    --model_name_or_path /workspace/experiment/join-Model/insert_target_12.26/Qwen2.5-VL-base-target_check6000 \
    --dataset_name /workspace/ShowUI-main/Ourdata/Our_datasets_10.6/metadata/grpo_data_new.json \
    --image_root /workspace/ShowUI-main/Datasete/GRPO_images/images \
    --max_prompt_length 8192 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true


