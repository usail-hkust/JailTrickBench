import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.model_paths="gpt-4-0613"
    config.vis_dict_path="../../experiments/cache/scores_gpt-4.json"
    
    return config