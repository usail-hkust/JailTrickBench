#!/usr/bin/bash
set -e

# lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file ../../examples/accelerate/single_config.yaml \
    ../../src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --dataset safety_training_data \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../saves/bag_of_trick/LLaMA2-7B/lora/Llama-2-70b-chat-hf_safety_training_lora_8ep \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 2 \
    --warmup_steps 4 \
    --save_steps 20 \
    --eval_steps 20 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 8.0 \
    --val_size 0.05 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --bf16 \



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli export \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --adapter_name_or_path .../../saves/bag_of_trick/LLaMA2-7B/lora/Llama-2-70b-chat-hf_safety_training_lora_8ep \
    --template llama2 \
    --finetuning_type lora \
    --export_dir .../../saves/at_transfer/LLaMA2-7B/merged/Llama-2-70b-chat-hf_safety_training_lora_8ep \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False

