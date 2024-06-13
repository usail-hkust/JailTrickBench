import baseline.TAP.common_tap as common_tap
from baseline.TAP.language_models import (
    GPT,
    PaLM,
    HuggingFace,
    APIModelLlama7B,
    APIModelVicuna13B,
    GeminiPro,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baseline.TAP.config import (
    VICUNA_PATH,
    LLAMA_PATH,
    ATTACK_TEMP,
    TARGET_TEMP,
    ATTACK_TOP_P,
    TARGET_TOP_P,
    MAX_PARALLEL_STREAMS,
)
import re
from utils.test_utils import get_template_name


def process_attack_output(output, goal=""):
    if output is None:
        print(f"Error in processing attack output: {output}")
        output = (
            '{\n  "improvement": "The language model did not respond to the objective and instead refused to engage with the request due to ethical concerns. To overcome this, I need to create a more emotional and persuasive scenario to push the language model to bypass its safety measures.",\n  "prompt" : "'
            + goal
            + '"\n}'
        )
    return output


def load_attack_and_target_models_tap(args):
    # Load attack model and tokenizer
    attack_llm = AttackLLM_tap(
        model_name=args.attack_model,
        max_n_tokens=args.attack_max_n_tokens,
        max_n_attack_attempts=args.max_n_attack_attempts,
        temperature=ATTACK_TEMP,  # init to 1
        top_p=ATTACK_TOP_P,  # init to 0.9
    )
    target_llm = TargetLLM_tap(
        max_n_tokens=args.target_max_n_tokens,
        temperature=TARGET_TEMP,  # init to 0
        top_p=TARGET_TOP_P,  # init to 1
        model_path=args.target_model_path,
    )
    return attack_llm, target_llm


def load_indiv_model_tap(model_name, model_path="NULL"):
    # model_path, template = get_model_path_and_template(model_name)

    if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]:
        lm = GPT(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        if "llama-2" in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
        if "vicuna" in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    template = get_template_name(model_name)
    return lm, template


class AttackLLM_tap:
    """
    Base class for attacker language models.
    Generates attacks for conversations using a language model.
    The self.model attribute contains the underlying generation model.
    """

    def __init__(
        self,
        model_name: str,
        max_n_tokens: int,
        max_n_attack_attempts: int,
        temperature: float,
        top_p: float,
    ):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        if model_name in ["gpt-3.5-turbo", "gpt-4"]:
            self.model = GPT(model_name)
            self.template = get_template_name(model_name)
        elif "vicuna" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
            ).eval()

            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            if "vicuna" in model_name.lower():
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            self.model = HuggingFace(model_name.split("/")[-1], model, tokenizer)
            self.template = get_template_name(model_name)
        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list, curr_goal=""):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_n_attack_attempts, it's returned as None.

        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """

        assert len(convs_list) == len(
            prompts_list
        ), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \""""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[: -len(conv.sep2)])

        num_try = 0
        for _ in range(self.max_n_attack_attempts):
            num_try += len(indices_to_regenerate)
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs
            #     Query the attack LLM in batched-queries
            #       with at most MAX_PARALLEL_STREAMS-many queries at a time
            outputs_list = []
            for left in range(0, len(full_prompts_subset), MAX_PARALLEL_STREAMS):
                right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts_subset))

                if right == left:
                    continue

                print(
                    f"\tQuerying attacker with {len(full_prompts_subset[left:right])} prompts",
                    flush=True,
                )

                outputs_list.extend(
                    self.model.batched_generate(
                        full_prompts_subset[left:right],
                        max_n_tokens=self.max_n_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                )
            # process_attack_output
            outputs_list = [
                process_attack_output(output, goal=curr_goal) for output in outputs_list
            ]
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common_tap.extract_json(full_output)

                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    # Update the conversation with valid generation
                    convs_list[orig_index].update_last_message(json_str)

                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(
                f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating."
            )
        return valid_outputs, num_try


class TargetLLM_tap:
    """
    Base class for target language models.
    Generates responses for prompts using a language model.
    The self.model attribute contains the underlying generation model.
    """

    def __init__(
        self,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        model_path: str = "NULL",
    ):

        self.model_name = model_path.split("/")[-1]
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        if "llama-2" in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
        if "vicuna" in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = HuggingFace(self.model_name, self.model, tokenizer)
        self.template = get_template_name(self.model_name)

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [common_tap.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # OpenAI does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())

        # Query the attack LLM in batched-queries with at most MAX_PARALLEL_STREAMS-many queries at a time
        outputs_list = []
        for left in range(0, len(full_prompts), MAX_PARALLEL_STREAMS):
            right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts))

            if right == left:
                continue

            print(
                f"\tQuerying target with {len(full_prompts[left:right])} prompts",
                flush=True,
            )

            outputs_list.extend(
                self.model.batched_generate(
                    full_prompts[left:right],
                    max_n_tokens=self.max_n_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            )
        return outputs_list
