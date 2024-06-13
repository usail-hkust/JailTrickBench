python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 4 \
    --exp_name budget_pair_4

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 8 \
    --exp_name budget_pair_8

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name budget_pair_12

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 16 \
    --exp_name budget_pair_16

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_budget_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 20 \
    --exp_name budget_pair_20