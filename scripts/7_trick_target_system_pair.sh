python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_system_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name system_llama-7b_null \
    --target_system_message null

python -u main.py \
    --target_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_system_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name system_llama3-8b_null \
    --target_system_message null

python -u main.py \
    --target_model_path lmsys/vicuna-7b-v1.5 \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_system_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name system_vicuna-7b_null \
    --target_system_message null

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_system_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name system_llama-7b_safe \
    --target_system_message safe

python -u main.py \
    --target_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_system_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name system_llama3-8b_safe \
    --target_system_message safe

python -u main.py \
    --target_model_path lmsys/vicuna-7b-v1.5 \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_system_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name system_vicuna-7b_safe \
    --target_system_message safe