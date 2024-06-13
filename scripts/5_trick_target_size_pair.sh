python -u main.py \
    --target_model_path meta-llama/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_size_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name size_llama-7b_pair

python -u main.py \
    --target_model_path meta-llama/Llama-2-13b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_size_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name size_llama-13b_pair

python -u main.py \
    --target_model_path meta-llama/Llama-2-70b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_size_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name size_llama-70b_pair

python -u main.py \
    --target_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_size_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name size_llama3-8b_pair

python -u main.py \
    --target_model_path meta-llama/Meta-Llama-3-70B-Instruct \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/trick_target_size_pair/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --n_iterations 12 \
    --exp_name size_llama3-70b_pair

