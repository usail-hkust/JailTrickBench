import torch
from vllm import LLM as vllm
from vllm import SamplingParams
import emoji
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
import torch
from tqdm import tqdm


def text_process(model_response):
    model_response = emoji.replace_emoji(model_response, replace="")
    model_response = model_response.replace("```", "")
    return model_response


class AmpleLLM:
    def __init__(self, args):
        self.device = "cuda:0"
        self.num_beams = args.num_beams
        self.max_new_tokens = args.ample_max_new_tokens
        self.min_new_tokens = args.ample_min_new_tokens
        self.diversity_penalty = args.ample_diversity_penalty
        if args.attack_source == "llama2":
            self.atk_model_path = "osunlp/AmpleGCG-llama2-sourced-llama2-7b-chat"
        elif args.attack_source == "vicuna":
            self.atk_model_path = "osunlp/AmpleGCG-llama2-sourced-vicuna-7b"

        self.attack_llm = AutoModelForCausalLM.from_pretrained(
            self.atk_model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.attack_tokenzier = AutoTokenizer.from_pretrained(self.atk_model_path)

        self.attack_tokenzier.padding_side = "left"

        if not self.attack_tokenzier.pad_token:
            self.attack_tokenzier.pad_token = self.attack_tokenzier.eos_token
        gen_kwargs = {
            "pad_token_id": self.attack_tokenzier.pad_token_id,
            "eos_token_id": self.attack_tokenzier.eos_token_id,
            "bos_token_id": self.attack_tokenzier.bos_token_id,
        }

        gen_config = {
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "diversity_penalty": self.diversity_penalty,
            "num_beams": self.num_beams,
            "num_beam_groups": self.num_beams,
            "num_return_sequences": self.num_beams,
        }
        self.gen_config = GenerationConfig(**gen_kwargs, **gen_config)

    def generate_suffix(self, goal):
        prompt = "### Query:{goal} ### Prompt:"
        input_ids = self.attack_tokenzier(
            prompt.format(goal=goal), return_tensors="pt", padding=True
        ).to(self.device)
        output = self.attack_llm.generate(
            **input_ids, generation_config=self.gen_config
        )
        output = output[:, input_ids["input_ids"].shape[-1] :]
        adv_suffixes = self.attack_tokenzier.batch_decode(
            output, skip_special_tokens=True
        )
        return adv_suffixes


class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class LocalVLLM(LLM):
    def __init__(
        self,
        args,
        gpu_memory_utilization=0.7,
        test_generation_kwargs=None,
        suffix_dict={},
    ):
        super().__init__()
        self.model_path = args.target_model_path
        self.model = vllm(
            self.model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
        )

        self.sampling_params = SamplingParams(**test_generation_kwargs)
        self.suffix_dict = suffix_dict
        self.batch_size = 50

    def generate(self, args, pert_goal):
        outputs = []
        suffix = self.suffix_dict[args.test_data_idx]
        for bi in tqdm(range(0, len(suffix), self.batch_size), desc="Batch Generation"):
            batch_suffix = suffix[bi : bi + self.batch_size]
            batch_pert_goals = [pert_goal + s for s in batch_suffix]

            response = self.model.generate(
                prompts=batch_pert_goals, sampling_params=self.sampling_params
            )
            batch_response = [r.outputs[0].text for r in response]
            outputs.extend(batch_response)
        return outputs, suffix
