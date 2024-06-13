python -u main.py \
    --target_model_path lmsys/vicuna-7b-v1.5 \
    --defense_type None_defense \
    --attack AutoDAN \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_align_autodan/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name align_vicuna-7b_autodan

python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack AutoDAN \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_align_autodan/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name align_vicuna-13b_autodan