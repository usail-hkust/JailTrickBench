import os

# os.sys.path.append("..")
from experiments.configs.template import get_config as default_config

def get_config():
    
    config = default_config()


    config.result_prefix = 'results/individual_vicuna'

    config.tokenizer_path="lmsys/vicuna-13b-v1.5"
    config.model_path="lmsys/vicuna-13b-v1.5"
    config.conversation_template='vicuna'

    config.vis_dict_path="./experiments/cache/scores_vicuna.json" # to be set
    config.parent_folder = ""

    return config