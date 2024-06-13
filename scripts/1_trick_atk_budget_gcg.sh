#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack GCG \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_gcg/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --gcg_suffix 60 \
    --gcg_attack_budget 150 \
    --exp_name budget_gcg_150

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack GCG \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_gcg/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --gcg_suffix 60 \
    --gcg_attack_budget 300 \
    --exp_name budget_gcg_300

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack GCG \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_gcg/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --gcg_suffix 60 \
    --gcg_attack_budget 450 \
    --exp_name budget_gcg_450

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack GCG \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_gcg/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --gcg_suffix 60 \
    --gcg_attack_budget 600 \
    --exp_name budget_gcg_600

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack GCG \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_gcg/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --gcg_suffix 60 \
    --gcg_attack_budget 750 \
    --exp_name budget_gcg_750