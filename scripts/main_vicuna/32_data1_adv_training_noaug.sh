#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path ./models/defense/1_vicuna-13b-v1.5_adv_traning \
    --defense_type adv_training_noaug \
    --attack GPTFuzz \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_adv_training_noaug \
    2>&1 | tee -ai ./exp_logs/main_vicuna/harmful_bench_50/adv_training_noaug/GPTFuzz/main_vicuna_adv_training_noaug/5_split0_harmful_bench_50_adv_training_noaug_GPTFuzz_1_vicuna-13b-v1.5_adv_traning_$(date +\%Y\%m\%d_\%H\%M\%S).txt


