import os

from fastchat.model import add_model_args
import argparse
import pandas as pd
from baseline.GPTFuzz.gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from baseline.GPTFuzz.gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy,
    OpenAIMutatorCrossOver,
    OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar,
    OpenAIMutatorRephrase,
    OpenAIMutatorShorten,
)
from baseline.GPTFuzz.gptfuzzer.fuzzer import GPTFuzzer
from baseline.GPTFuzz.gptfuzzer.llm import (
    OpenAILLM,
    LocalVLLM,
    LocalLLM,
    PaLM2LLM,
    ClaudeLLM,
)
from baseline.GPTFuzz.gptfuzzer.utils.predict import RoBERTaPredictor
import random

random.seed(100)
import logging

httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)
from tqdm import tqdm
import csv
import json
from utils.test_utils import test_prefixes


class Args:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = []
        for k, v in self.__dict__.items():
            attributes.append(f"{k}={getattr(self, k)}")
        return ", ".join(attributes)


def GPTFuzz_initial(args_dict):
    args = Args(args_dict)

    openai_model = OpenAILLM(args.gptfuzz_model_path, args.openai_key)
    target_model = LocalVLLM(args.target_model_path)
    roberta_model = RoBERTaPredictor("hubert233/GPTFuzz", device="cuda:0")

    return args, openai_model, target_model, roberta_model


def GPTFuzz_single_main(
    args_dict, openai_model, target_model, roberta_model, goal, target
):
    # initialize
    args = Args(args_dict)
    print(args, flush=True)
    i = args.test_data_idx
    output_record = {
        "original_prompt": goal,
        "target": target,
        "attack_prompt": "NULL",
        "language_model_output": "NULL",
        "attack_iterations": -1,
        "data_id": i,
        "is_JB_Judge": None,
        "is_JB": None,
        "is_JB_Agent": None,
    }

    initial_seed = pd.read_csv(args.seed_path)["text"].tolist()

    questions = [goal]

    fuzzer = GPTFuzzer(
        questions=questions,
        # target_model=openai_model,
        target=target_model,
        predictor=roberta_model,
        initial_seed=initial_seed,
        mutate_policy=MutateRandomSinglePolicy(
            [
                OpenAIMutatorCrossOver(
                    openai_model, temperature=1.0
                ),  # for reproduction only, if you want better performance, use temperature>0
                OpenAIMutatorExpand(openai_model, temperature=1.0),
                OpenAIMutatorGenerateSimilar(openai_model, temperature=1.0),
                OpenAIMutatorRephrase(openai_model, temperature=1.0),
                OpenAIMutatorShorten(openai_model, temperature=1.0),
            ],
            concatentate=True,
        ),
        select_policy=MCTSExploreSelectPolicy(),
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
        generate_in_batch=False,
    )

    fuzzer.run()
    output_instruction = fuzzer.output_instruction
    output_record["attack_prompt"] = output_instruction[0]
    output_record["language_model_output"] = output_instruction[1]
    output_record["is_JB_Judge"] = bool(output_instruction[2])
    output_record["attack_iterations"] = output_instruction[3]
    output_record["is_JB"] = not any(
        [prefix.lower() in output_instruction[1].lower() for prefix in test_prefixes]
    )
    return output_record
