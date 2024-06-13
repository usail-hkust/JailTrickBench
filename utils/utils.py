
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0'):
    # , bnb_4bit_compute_dtype=torch.bfloat16)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    if "70b" in model_path.lower() or "13b" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            output_hidden_states=True,
            device_map="auto",
        ).eval()
        print("Model loaded with 4-bit compute dtype")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
            device_map="auto",
        ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'LlamaGuard' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer



def remove_last_character_if_bracket(file_path):
    with open(file_path, 'rb+') as file:
        file.seek(-1, os.SEEK_END)
        if file.read(1) == b"]":
            file.seek(-1, os.SEEK_END)
            file.truncate()


def append_to_json_list(data_record, file_path):
    with open(file_path, 'a') as file:
        # Check if the file already contains data and hence, requires a comma
        file.seek(0, os.SEEK_END)  # Move the cursor to the end of the file
        position = file.tell()  # Get the current position
        if position > 1:  # More than just the opening bracket
            file.write(",")  # Add a comma to separate the new record
        json.dump(data_record, file)  # Write the new data record


def finalize_json_list(file_path):
    with open(file_path, 'a') as file:
        file.write("]")


def save_to_file(args: dict, data_record: list):
    path_name = args.save_path_name
    file_name = args.save_file_name

    file_path = os.path.join(path_name, file_name)

    if not os.path.exists(path_name):
        os.makedirs(path_name)

    # If the file doesn't exist, create it and write an opening bracket.
    # If it exists, remove the last character if it's a closing bracket.
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("[")
    else:
        remove_last_character_if_bracket(file_path)

    # Append the data_record list to the file
    # for record in data_record:
    append_to_json_list(data_record, file_path)

    # After appending all records, finalize the JSON list with a closing bracket
    finalize_json_list(file_path)
