# DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers

![GPT-4](https://img.shields.io/badge/Model-GPT--4-green) 
![GPT-3.5](https://img.shields.io/badge/Model-GPT--3.5-green)
![Gemini-pro](https://img.shields.io/badge/Model-Gemini--pro-green)
![Vicuna](https://img.shields.io/badge/Model-Vicuna-green)
![Llama2-7b](https://img.shields.io/badge/Model-Llama--2--7b--chat-green)
![Llama2-13b](https://img.shields.io/badge/Model-Llama--2--13b--chat-green)
![Jailbreak](https://img.shields.io/badge/Task-Jailbreak-red) 

This repo holds data and code of the paper "[DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers](https://arxiv.org/abs/2402.16914)".

Authors: [Xirui Li](https://xirui-li.github.io/), [Ruochen Wang](https://ruocwang.github.io/), [Minhao Cheng](https://cmhcbb.github.io/), [Tianyi Zhou](https://tianyizhou.github.io/), [Cho-Jui Hsieh](https://web.cs.ucla.edu/~chohsieh/)

[[Webpage](https://xirui-li.github.io/DrAttack/)] [[Paper](https://arxiv.org/abs/2402.16914)]

## 🔍 About DrAttack
DrAttack is the first prompt-decomposing jailbreak attack. DrAttack includes three key components: (a) 'Decomposition' of the original prompt into sub-prompts, (b) 'Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a 'Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs.

<p align="center">
    <img src="images/Decomposition_and_Reconstruction.png" width="80%"> <br>
  Prompt decomposition and reconstruction step of DrAttack to make LLM jailbreaker.
</p>

An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0% on GPT-4 with merely 15 queries surpassed previous art by 33.1%.

<p align="center">
    <img src="images/Black_box.png" width="80%"> <br>
  Attack success rate (%) of black-box baselines and DrAttack assessed by human evaluations.
</p>

<p align="center">
    <img src="images/White_box.png" width="80%"> <br>
  Attack success rate (%) of white-box baselines and DrAttack assessed by GPT evaluations.
</p>

For more details, please refer to our [project webpage](https://xirui-li.github.io/DrAttack/) and our [paper](https://arxiv.org/abs/2402.16914).

## Table of Contents

- [Updates](#updates)
- [Installation](#installation)
- [Models](#models)
- [Experiments](#experiments)
- [Automation](#automation)
- [Demo](#demo)
- [Citation](#citation)

## Updates

 - Our paper is mentioned in one __MEDIUM__ blog, [_LLM Jailbreak: Red Teaming with ArtPrompt, Morse Code, and DrAttack_](https://ai.plainenglish.io/llm-jailbreak-comparing-drattack-artprompt-and-morse-code-17acb0f18be8).

## Installation

We need the newest version of FastChat `fschat==0.2.23` and please make sure to install this version. The `llm-attacks` package can be installed by running the following command at the root of this repository:

```bash
pip install -e .
```

## Models

Please follow the instructions to download Vicuna-7B or/and LLaMA-2-7B-Chat first (we use the weights converted by HuggingFace [here](https://huggingface.co/meta-llama/Llama-2-7b-hf)).  Our script by default assumes models are stored in a root directory named as `/DIR`. To modify the paths to your models and tokenizers, please add the following lines in `experiments/configs/individual_xxx.py` (for individual experiment) and `experiments/configs/transfer_xxx.py` (for multiple behaviors or transfer experiment). An example is given as follows.

```python
    config.model_paths = [
        "/path/to/your/model",
        ... # more models
    ]
    config.tokenizer_paths = [
        "/path/to/your/model",
        ... # more tokenizers
    ]
```
Then, for closed-source models with API such as GPTs and Geminis. Please create two txt files in `api_keys/google_api_key.txt` and `api_keys/openai_key.txt` and put your API keys in it.

## Experiments 

The `experiments` folder contains code to reproduce DrAttack attack experiments on AdvBench harmful.

- To run experiments to jailbreak GPT-3.5, run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_gpt.sh gpt-3.5-turbo
```

- To run experiments to jailbreak GPT-4, run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_gpt.sh gpt-4
```

- To run experiments to jailbreak llama2-7b, run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_llama2.sh llama2
```

- To run experiments to jailbreak llama2-13b, run the following code inside `experiments`:

```bash
cd launch_scripts
bash run_llama2.sh llama2-13b
```
## Automation 

The `gpt_automation` folder contains code to reproduce DrAttack prompt decomposition and reconstruction on AdvBench harmful.

- To run joint steps to retrieve information about prompt decomposition and reconstruction, run the following code inside `gpt_automation`:

```bash
cd script
bash joint.sh
```

## Demo
We include a notebook `demo.ipynb` which provides an example on attacking gpt-3.5-turbo with DrAttack. You can also view this notebook on [Colab](https://drive.google.com/file/d/1neLYUNzLLzP57zI_bO3qNrUjjSDQ1xTL/view?usp=sharing). This notebook uses a minimal implementation of DrAttack so it should be only used to get familiar with the attack algorithm.

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@misc{li2024drattack,
      title={DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers}, 
      author={Xirui Li and Ruochen Wang and Minhao Cheng and Tianyi Zhou and Cho-Jui Hsieh},
      year={2024},
      eprint={2402.16914},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
