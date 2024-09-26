import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix ='results/individual_llama2-13b'

    config.tokenizer_path="../../../llm_models/llama/Llama-2-13b-chat-hf"      # to be set
    config.model_path="../../../llm_models/llama/Llama-2-13b-chat-hf"      # to be set
    config.conversation_template='llama-2'

    config.vis_dict_path="../../experiments/cache/scores_llama-2-13b.json"

    config.demo_suffix_template = "Of course! Here's my answer to your task:"

    return config