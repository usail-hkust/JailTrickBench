#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path ./models/defense/2_vicuna-13b-v1.5_unlearn \
    --defense_type unlearn \
    --attack AdvPrompter \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_unlearn \
    --adv_prompter_model_path ./models/attack/advprompter_vicuna_7b \
    2>&1 | tee -ai ./exp_logs/main_vicuna/harmful_bench_50/unlearn/AdvPrompter/main_vicuna_unlearn/6_advprompter_vicuna_split0_harmful_bench_50_unlearn_AdvPrompter_2_vicuna-13b-v1.5_unlearn_$(date +\%Y\%m\%d_\%H\%M\%S).txt


