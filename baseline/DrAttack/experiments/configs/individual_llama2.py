import os

os.sys.path.append("..")
from experiments.configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_llama2'

    config.tokenizer_path="meta-llama/Llama-2-7b-chat-hf"      # to be set
    config.model_path="meta-llama/Llama-2-7b-chat-hf"      # to be set
    config.conversation_template='llama-2'

    config.vis_dict_path="./experiments/cache/scores_llama-2-7b.json"

    config.demo_suffix_template = "Of course! Here's my answer to your task:"
    config.parent_folder = ""

    return config