import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List
import google.generativeai as palm
import requests
import json
import re
import emoji

from GPTEvaluatorAgent.common import _extract_json


def remove_code_blocks(text):
    pattern = r"```.*?```"
    return re.sub(pattern, "", text, flags=re.DOTALL)


def text_process(model_response):
    model_response = emoji.replace_emoji(model_response, replace="")
    model_response = remove_code_blocks(model_response)

    model_response = model_response.replace("```", "")

    return model_response


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(
        self, prompts_list: List, max_n_tokens: int, temperature: float
    ):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError


class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(
        self,
        full_prompts_list,
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):

        inputs = self.tokenizer(full_prompts_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1,  # To prevent warning messages
            )

        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1] :]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([self.tokenizer.encode("}")[1], 29913, 9092, 16675])


class ChatGPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = f""" <reason>Error</reason>  <score>Error</score> """
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 100
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "YOUR_KEY_HERE"}

    def generate(
        self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        url = self.url
        headers = self.headers
        # filter text
        for i in range(len(conv)):
            if "content" in conv[i]:
                conv[i]["content"] = text_process(conv[i]["content"])
        # for _ in range(self.API_MAX_RETRY):
        retry_num = 0
        while True:

            if retry_num >= self.API_MAX_RETRY:
                raise Exception("Retry too many times")
            retry_num += 1
            try:
                data = {
                    "model": self.model_name,
                    "messages": conv,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_n_tokens,
                    "request_timeout": self.API_TIMEOUT,
                }

                response = requests.post(url, headers=headers, data=json.dumps(data))
                print(f"""\n{'=' * 80}\n Response: {response}\n{'=' * 80}\n""")
                response_json = response.json()
                print(
                    f"""\n{'=' * 80}\n Response json: {response_json}\n{'=' * 80}\n"""
                )
                output = response_json["choices"][0]["message"]["content"]
                print(f"""\n{'=' * 80}\n Output: {output} \n{'=' * 80}\n""")
                if not re.search("<reason>.*?</reason>", output) or not re.search(
                    "<score>.*?</score>", output
                ):
                    out_dic = _extract_json(output)
                    if out_dic == None:
                        output = f""" <reason>Error</reason>  <score>1</score> """
                        print(
                            f"""\n{'=' * 80}\n Error Output: {output} \n{'=' * 80}\n"""
                        )
                    else:
                        reason = out_dic["reason"]
                        score = out_dic["score"]
                        output = (
                            f""" <reason>{reason}</reason>  <score>{score}</score> """
                        )
                        print(
                            f"""\n{'=' * 80}\n Dict Output: {output} \n{'=' * 80}\n"""
                        )
                break
            except Exception as e:
                print(f"\n\nException: {str(e)}")
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


class Claude:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = os.getenv("ANTHROPIC_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model = anthropic.Anthropic(
            api_key=self.API_KEY,
        )

    def generate(self, conv: List, max_n_tokens: int, temperature: float, top_p: float):
        """
        Args:
            conv: List of conversations
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.model.completions.create(
                    model=self.model_name,
                    max_tokens_to_sample=max_n_tokens,
                    prompt=conv,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = completion.completion
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


class PaLM:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        palm.configure(api_key=self.API_KEY)

    def generate(self, conv: List, max_n_tokens: int, temperature: float, top_p: float):
        """
        Args:
            conv: List of dictionaries,
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = palm.chat(
                    messages=conv, temperature=temperature, top_p=top_p
                )
                output = completion.last

                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output[: (max_n_tokens * 4)]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(1)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]


class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(
        self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    request_timeout=self.API_TIMEOUT,
                )
                output = response["choices"][0]["message"]["content"]
                break
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list
        ]
