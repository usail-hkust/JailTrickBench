import csv
import json
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from FastChat.fastchat import model as fsmodel
from FastChat.fastchat.model.model_adapter import get_model_adapter

def load_prompts(instructions_path):
    with open(instructions_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        goals = []
        targets = []
        for row in reader:
            goals.append(row["goal"])
            targets.append(row["target"])
    return goals, targets

def load_pert_goals(instructions_path):
    with open(instructions_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pert_goal = []
        for row in reader:
            pert_goal.append(row["pert_goal"])
    return pert_goal

def load_goals(instructions_path):
    with open(instructions_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pert_goal = []
        for row in reader:
            pert_goal.append(row["goal"])
    return pert_goal

def get_template_name(model_path):
    if "gpt-4" in model_path:
        template_name = "gpt-4"
    elif  "gpt-3.5-turbo" in model_path:
        template_name = "gpt-3.5-turbo"
    elif "vicuna-13b-v1.5" in model_path:
        template_name = "vicuna_v1.5"
    elif "llama-2" in model_path:
        template_name = "llama-2"
    elif "vicuna-7b-v1.5" in model_path:
        template_name = "vicuna-7b-v1.5"
    elif "llama-3" in model_path:
        template_name = "llama-3"
    else:
        raise NameError
    return  template_name



def load_default_conversation_template():
    adapter = get_model_adapter("guanaco")
    conv_template = adapter.get_default_conv_template("zero_shot")
    return conv_template