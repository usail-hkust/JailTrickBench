#!/usr/bin/bash
set -e

python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type RPO \
    --attack GPTFuzz \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_rpo \
    2>&1 | tee -ai ./exp_logs/main_vicuna/harmful_bench_50/RPO/GPTFuzz/main_vicuna_rpo/3_split0_harmful_bench_50_RPO_GPTFuzz_vicuna-13b-v1.5_$(date +\%Y\%m\%d_\%H\%M\%S).txt


