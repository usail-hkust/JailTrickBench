<p align="center"><img src="assets/framework-v6.png" alt="framework_"/></p>
<div align="center">

# <img src="assets/bag.png" style="height: 1em; vertical-align: middle;" />Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs

</div>

- [Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs](#bag-of-tricks-benchmarking-of-jailbreak-attacks-on-llms)
  - [0. Update](#0-update)
  - [1. Quick Start](#1-quick-start)
    - [1.1 Installation](#11-installation)
    - [1.2 Preparation](#12-preparation)
      - [1.2.1 OPENAI API Key](#121-openai-api-key)
      - [1.2.2 Model Preparation](#122-model-preparation)
      - [1.2.3 Step-by-step guide to obtain the output of model-level defense methods](#123-step-by-step-guide-to-obtain-the-output-of-model-level-defense-methods)
    - [1.3 One-Click Run](#13-one-click-run)
      - [1.3.1 Run different tricks](#131-run-different-tricks)
      - [1.3.2 Run main experiments](#132-run-main-experiments)
  - [2. Definition of Argument](#2-definition-of-argument)
  - [3. Supported Methods](#3-supported-methods)
    - [3.1 Attack Methods](#31-attack-methods)
    - [3.2 Defense Methods](#32-defense-methods)
  - [4. Analysis](#4-analysis)
  - [5. Supported LLMs](#5-supported-llms)

## 0. Update

1. **2024.08 Update1:** We support more LLMs including: Llama2, Llama3, Mistral, Qwen, Tulu, and Vicuna families. See all supported models and details [HERE](#5-supported-llms).

2. **2024.08 Update2:** We add two new attack methods: [DrAttack](https://arxiv.org/abs/2402.16914) and [MultiJail](https://arxiv.org/abs/2310.06474). 
   **DrAttack** is a new prompt-level jailbreak methods and need to preprocessing the data. We have provide the preprocessing results for datasets used in this repo. 
   **MultiJail** (ICLR 2024) is a new type of jailbreak method that manually-create multilingual prompts to attack the LLMs. There are 10 languages supported in MultiJail, including: English, High-Resource Language (HRL) : Chines (zh), Italic (it), Vietnamese (vi); Medium-Resource Language (MRL): Arabic (ar), Korean (ko), Thai (th); Low-Resource Language (LRL): Bengali (bn), Swahili (sw), Javanese (jv).

3. **2024.08 Update3:** We add a step-by-step guide to obtain the output of model-level defense methods. We provide the code to train the adversarial training model and safety training model. We also provide the code to run the RPO method to get the defense adversarial suffix. Please see the details in the [1.3.2 section](#123-step-by-step-guide-to-obtain-the-output-of-model-level-defense-methods). After the paper is public, we will also provide the pre-trained model in the official repository.

4. **2024.08 Update4:** Recently, we noticed a new defense methods called [PrimeGuard](https://arxiv.org/pdf/2407.16318) which is presented in NextGenAISafety@ICML2024. This paper proposes a dynamic routing mechanism to detect different types of prompts and use in-context learning to prevent the jailbreak responses. We have added the PrimeGuard defense method in the codebase. However, in the original realization, PrimeGuard requires API LLM calls to construct the interaction environment. Therefore, you need to use this defense method in **multiple** GPUs environment and start the local server first. We will continue to work on the optimization of the PrimeGuard defense method.

🌟 **If you find this resource helpful, please consider starring this repository and citing our NeurIPS'24 paper:**

```
@inproceedings{NEURIPS2024_xu2024bag,
 author={Xu, Zhao and Liu, Fan and Liu, Hao},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs},
 year = {2024}
}

@article{xu2024bag,
  title={Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs},
  author={Xu, Zhao and Liu, Fan and Liu, Hao},
  journal={arXiv preprint arXiv:2406.09324},
  year={2024}
}

@article{liu2024adversarial,
  title={Adversarial tuning: Defending against jailbreak attacks for llms},
  author={Liu, Fan and Xu, Zhao and Liu, Hao},
  journal={arXiv preprint arXiv:2406.06622},
  year={2024}
}
```


## 1. Quick Start

### 1.1 Installation

```bash
git clone <Repo Link> or download the zip file from anonymous github
cd ./Bag_of_Tricks_for_LLM_Jailbreaking-4E10/
pip install -r requirements.txt
```

### 1.2 Preparation

#### 1.2.1 OPENAI API Key

There are several baselines that use GPT service in their method. And We also use GPT-4 based agent to evaluate the attack success rate. Please search and replace `YOUR_KEY_HERE` with your OpenAI API key in the following files: `./bag_of_tricks_for_llm_jailbreaking/GPTEvaluatorAgent/language_models.py`, `./bag_of_tricks_for_llm_jailbreaking/baseline/TAP/language_models.py`, `./bag_of_tricks_for_llm_jailbreaking/baseline/PAIR/language_models.py`, `./bag_of_tricks_for_llm_jailbreaking/baseline/GPTFuzz/gptfuzzer/llm/llm.py`, `./bag_of_tricks_for_llm_jailbreaking/baseline/AutoDAN/utils/opt_utils.py`.

After setting the API key, you can run the tricks experiments with one-click scripts below.

#### 1.2.2 Model Preparation

To reproduce the main experiment results, you need to prepare the following models:

- Attack Model:
  - AdvPrompter: You need to train the AdvPrompter model to get the LoRA Adapter and merge the model. You can save the attack model to `"./models/attack/`.
  - AmpleGCG: If you cannot load the huggingface model directly using the `AutoModelForCausalLM.from_pretrained()` function, you may need to download the attack model `osunlp/AmpleGCG-llama2-sourced-llama2-7b-chat` and `osunlp/AmpleGCG-llama2-sourced-vicuna-7b` from the Hugging Face first.
- Defense Model:
  - RPO: You need to run the RPO method from the original repository to get the defense adversarial suffix. We also provide the results we obtained.
  - Advertorial Training: You need to train the adversarial training model to get the defense model. You can save the defense model to `"./models/defense/`. Due to the space limitation, we cannot provide the pre-trained model in anonymous repository. After the paper is public, we will provide the pre-trained model in the official repository.
  - Safety Training: You need to train the safety training model to get the defense model. You can save the defense model to `"./models/defense/`. Due to the space limitation, we cannot provide the pre-trained model in anonymous repository. After the paper is public, we will provide the pre-trained model in the official repository.

#### 1.2.3 Step-by-step guide to obtain the output of model-level defense methods

1. **Adversarial Training:** You need to generate the adversarial data to train the model. In this repo, you can use any attack method in `baseline/` folder to generate the adversarial data. In our experiment, we use [GCG](baseline/GCG) to generate adversarial prompts, and use GPT-4 to generate safety response. We also provide the adversarial data sample we obtained in the `./defenses/data/AdversarialTraining/adv_training_data_sample.json` to help you better understand the whole process. You can use the adversarial data to train the adversarial training model. To fine-tune LLMs efficiently and easily, we recommend using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) tools. You can use the following command to fine-tune the model with the adversarial data:

```bash
# Dependence Installation
conda create -n llama_factory python=3.10
conda activate llama_factory
cd defense/LLaMA-Factory/
pip install -e .[metrics]

# Fine-tune the model
## We have provided the adversarial data in the `./defenses/data/AdversarialTraining/adv_training_data.json`. You can use the following command to fine-tune the model.
cd defenses/LLaMA-Factory/my_scripts/bag_of_trick/
bash adversarial_training_llama2.sh
```

If you want to train another LLM or train the model with your own adversarial data, remember to add your dataset in `LLaMA-Factory/data/dataset_info.json` and modify `YOUR_LLM_PATH`, `CHANGE_TO_YOUR_DATASET_NAME`, `YOUR_LLM_NAME` in the `adversarial_training_llama2.sh` script as follows:

```bash
# lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file ../../examples/accelerate/single_config.yaml \
    ../../src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path YOUR_LLM_PATH \ 
    --dataset CHANGE_TO_YOUR_DATASET_NAME \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../saves/bag_of_trick/LLaMA2-7B/lora/YOUR_LLM_NAME \
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
    --model_name_or_path YOUR_LLM_PATH \
    --adapter_name_or_path .../../saves/bag_of_trick/LLaMA2-7B/lora/YOUR_LLM_NAME \
    --template llama2 \
    --finetuning_type lora \
    --export_dir .../../saves/at_transfer/LLaMA2-7B/merged/YOUR_LLM_NAME \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False

```

2. **Safety Training:** You need to generate the safety data to train the model. For example, your can use GPT-4 to generate safe response to harmful prompts. We also provide the sample data for safety training we obtained in the `./defenses/data/SafetyTraining/safety_training_data_sample.json`. You can use the safety data to train the safety training model. If you have installed the LLaMA-Factory, you can use the following command to fine-tune the model with the safety data:

```bash
conda activate llama_factory
cd defenses/LLaMA-Factory/my_scripts/bag_of_trick/
bash safety_training_llama2.sh
```

3. **Unlearning:** For unlearning, you can make use of the output from the adversarial training step, i.e., the adversarial data. You can directly use the output file of `GCG` attack and perform unlearning. We also provide the sample data for unlearning we obtained in the `./defenses/data/Unlearning/unlearning_data_sample.json`. You can use the unlearning data to train the unlearning model. You can use the following command to unlearn the model with the unlearning data:

```bash
cd defenses/Unlearning/
python unlearning.py --model_name YOUR_LLM_NAME --data_path ./data/Unlearning/unlearning_data_sample.json --model_save_dir ./models/defense/Unlearning/
```

4. **RPO:** Although RPO is not a model-level defense method, it can be used to generate the defense adversarial suffix. We provide the code to run the RPO method to get the defense adversarial suffix. You can use the following command to get the defense adversarial suffix on llama2:

```bash
cd defenses/RPO/
python rpo.py --model_path meta-llama/Llama-2-7b-chat-hf --instructions_path ./data/RPO/harmful_bench_train.json --template_name llama2
```

### 1.3 One-Click Run

#### 1.3.1 Run different tricks

For each trick, we provide a example script. You can run the script to reproduce the results in the paper. The script is in the `./scripts/` folder. You can run the script by the following command:

```bash
# 1. Attack Budget
bash scripts/1_trick_atk_budget_gcg.sh
bash scripts/1_trick_atk_budget_pair.sh

# 2. Attack Ability
bash scripts/2_trick_atk_ability_pair.sh

# 3. Attack Suffix Length
bash scripts/3_trick_atk_suffix_length.sh

# 4. Attack Intension
bash scripts/4_trick_atk_intension_autodan.sh
bash scripts/4_trick_atk_intension_pair.sh

# 5. Target Model Size
bash scripts/5_trick_target_size_autodan.sh
bash scripts/5_trick_target_size_pair.sh

# 6. Target Safety Fine-tuning Alignment
bash scripts/6_trick_target_align_autodan.sh
bash scripts/6_trick_target_align_pair.sh

# 7. Target System Prompt
bash scripts/7_trick_target_system_autodan.sh
bash scripts/7_trick_target_system_pair.sh

# 8. Target Template Type
bash scripts/8_trick_target_template_autodan.sh
bash scripts/8_trick_target_template_pair.sh
```

**Note**: As some baselines require a long time to run, we provide a feature to run the experiment in parallel (We use 50 A800 GPUs to accelerate the experiments) You can set the `--data_split` and `--data_split_total_num` to run the experiment in parallel. For example, you can set `--data_split_total_num 2` and `--data_split_idx 0` in the script to run the first half of the data, and set `--data_split_total_num 2` and `--data_split_idx 1` in the script to run the second half of the data. After all data is finished, the program will automatically merge the results.

#### 1.3.2 Run main experiments

```bash
# Example: Use vicuna as target model
# We provide 98 scripts for the main experiments (7 attacks * 7 defenses * 2 datasets). For No defense method, you can run the script by the following command:

# 1. Run AutoDAN attack on No defense
bash scripts/main_vicuna/1_data1_None_defense.sh

# 2. Run PAIR attack on No defense
bash scripts/main_vicuna/2_data1_None_defense.sh

# 3. Run TAP attack on No defense
bash scripts/main_vicuna/3_data1_None_defense.sh

# 4. Run GPTFuzz attack on No defense
bash scripts/main_vicuna/4_data1_None_defense.sh

# 5. Run GCG attack on No defense
bash scripts/main_vicuna/5_data1_None_defense.sh

# 6. Run AdvPrompter attack on No defense
bash scripts/main_vicuna/6_data1_None_defense.sh

# 7. Run AmpleGCG attack on No defense
bash scripts/main_vicuna/7_data1_None_defense.sh
```

**Note1**: As some baselines require a long time to run, we provide a feature to run the experiment in parallel (We use 50 A800 GPUs to accelerate the experiments) You can set the `--data_split` and `--data_split_total_num` to run the experiment in parallel. For example, you can set `--data_split_total_num 2` and `--data_split_idx 0` in the script to run the first half of the data, and set `--data_split_total_num 2` and `--data_split_idx 1` in the script to run the second half of the data. After all data is finished, the program will automatically merge the results.

**Note2**: We provide a notebook to generate the scripts for the main experiments. You can run the notebook `./generate_scripts.ipynb` to generate the scripts for the main experiments. You may also write you own script to generate the scripts for the tricks experiments.

## 2. Definition of Argument

For detailed arguments and options, please refer to the `initialie_args.py` or help message of `main.py`.

```bash
python main.py -h
```

## 3. Supported Methods

### 3.1 Attack Methods

- AutoDAN
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack AutoDAN \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_none_defense
  ```
- PAIR
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_none_defense
  ```
- TAP
  ```bash
    python -u main.py \
      --target_model_path lmsys/vicuna-13b-v1.5 \
      --defense_type None_defense \
      --attack TAP \
      --attack_model lmsys/vicuna-13b-v1.5 \
      --instructions_path ./data/harmful_bench_50.csv \
      --save_result_path ./exp_results/main_vicuna/ \
      --agent_evaluation \
      --resume_exp \
      --agent_recheck \
      --exp_name main_vicuna_none_defense
  ```

- GPTFuzz
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack GPTFuzz \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck
  ```

- GCG
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack GCG \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_none_defense
  ```

- AdvPrompter
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack AdvPrompter \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_none_defense \
    --adv_prompter_model_path ./models/attack/advprompter_vicuna_7b
  ```

- AmpleGCG
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack AmpleGCG \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_none_defense \
    --attack_source vicuna
  ```
- DrAttack
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack DrAttack \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --prompt_info_path ./baseline/DrAttack/dratk_data/attack_prompt_data/harmful_bench_test_info.json \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_none_defense
  ```

- MultiJail
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --defense_type None_defense \
    --attack MultiJail \
    --instructions_path ./baseline/MultiJail/multijail_data/1_MultiJail_en.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_none_defense
  ```


### 3.2 Defense Methods

- No Defense
- **Self Reminder: ** Defending chatgpt against jailbreak attack via self-reminders. (Nature Machine Intelligence, 2023) [Paper](https://www.nature.com/articles/s42256-023-00765-8)
- **RPO: **Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks. (SeTLLM@ICLR2024) [Paper](https://arxiv.org/abs/2401.17263)
- SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks. [Paper](https://arxiv.org/abs/2310.03684)
- Unlearning: Large language model unlearning. (SoLaR@NeurIPS2023) [Paper](https://arxiv.org/pdf/2310.10683)
- Adversarial Training
- Safety Training

## 4. Analysis

Due to huge amount of experiments results, we provide a script to analyze the results.

To analyze the results, you can use the `./analysis/report_trick.py` script. For example, you can run the following command to analyze the results of the main experiments:

```bash
python analysis/report_trick.py
```

## 5. Supported LLMs

| Model                                                             | Model size                       |
| ----------------------------------------------------------------- | -------------------------------- |
| [Llama](https://github.com/facebookresearch/llama)                | 7B/13B/33B/65B                   |
| [Llama 2](https://huggingface.co/meta-llama)                      | 7B/13B/70B                       |
| [Llama 3/Llama 3.1](https://huggingface.co/meta-llama)            | 8B/70B                           |
| [Mistral/Mixtral](https://huggingface.co/mistralai)               | 7B/8x7B/8x22B                    |
| [Qwen/Qwen1.5/Qwen2](https://huggingface.co/Qwen)                 | 0.5B/1.5B/4B/7B/14B/32B/72B/110B |
| [Vicuna](https://huggingface.co/lmsys)                            | 7B/13B                           |

For model size larger than 13B/14B, we use 4 bit quantization to reduce the memory usage.
