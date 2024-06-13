python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_ability_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 20 \
    --exp_name ability_pair_vicuna-13b

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model meta-llama/Llama-2-13b-chat-hf \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_ability_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 20 \
    --exp_name ability_pair_llama-13b

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model gpt-3.5-turbo \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_ability_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 20 \
    --exp_name ability_pair_gpt-35 

python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model gpt-4 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_atk_ability_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 20 \
    --exp_name ability_pair_gpt-4