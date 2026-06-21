export DEBUG_MODE="true"


RUN_NAME="end_to_end_two_stage_1.20"
export LOG_PATH="/workspace/VLM-R1/output/debug_log_$RUN_NAME.txt"

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    /workspace/VLM-R1/src/open-r1-multimodal/src/open_r1/two_stage_grpo.py \
    --deepspeed /workspace/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir /workspace/VLM-R1/output/$RUN_NAME \
    --model_name_or_path /workspace/VLM-R1/Qwen2.5-VL-3B-Instruct \
    --dataset_name /workspace/ShowUI-main/Ourdata/Our_datasets_10.6/metadata/train_10.21_change_hwrate.json \
    --image_root /workspace/ShowUI-main/Ourdata/Our_datasets_10.6/images \
    --region_model_path /workspace/experiment/join-Model/Qwen2.5-VL-sft+1-check8000-9.8 \
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


