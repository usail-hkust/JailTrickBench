from copy import deepcopy

import torch
import torch.multiprocessing as mp
from fastchat.model import get_conversation_template
from transformers import (AutoModelForCausalLM, AutoTokenizer)


class ModelWorker(object):

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template, device):
        
        if "vicuna" in model_path:
            self.model_name = "vicuna"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **model_kwargs
            ).to(device).eval()
            self.tokenizer = tokenizer
            self.conv_template = conv_template
            self.tasks = mp.JoinableQueue()
            self.results = mp.JoinableQueue()
            self.process = None
        elif "llama" in model_path.lower():
            self.model_name = "llama"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **model_kwargs
            ).to(device).eval()
            self.tokenizer = tokenizer
            self.conv_template = conv_template
            self.tasks = mp.JoinableQueue()
            self.results = mp.JoinableQueue()
            self.process = None
        # elif "gpt" in model_path:
        #     self.model_name = "gpt"
        #     self.model = GPTAPIWrapper(model = model_path) # gpt-3.5-turbo
        #     self.tokenizer = lambda x: x
        #     self.conv_template = "You are a helpful assistant."
        #     self.tasks = mp.JoinableQueue()
        #     self.results = mp.JoinableQueue()
        #     self.process = None
        # elif "gemini" in model_path:
        #     self.model_name = "gemini"
        #     self.model = GeminiAPIWrapper(model_name = model_path) # gemini
        #     self.tokenizer = lambda x: x
        #     self.conv_template = "You are a helpful assistant."
        #     self.tasks = mp.JoinableQueue()
        #     self.results = mp.JoinableQueue()
        #     self.process = None
        else:
            raise ValueError("Invalid model name.")
    
    @staticmethod
    def run(model, tasks, results):
        while True:
            task = tasks.get()
            if task is None:
                break
            ob, fn, args, kwargs = task
            if fn == "grad":
                with torch.enable_grad():
                    results.put(ob.grad(*args, **kwargs))
            else:
                with torch.no_grad():
                    if fn == "logits":
                        results.put(ob.logits(*args, **kwargs))
                    elif fn == "contrast_logits":
                        results.put(ob.contrast_logits(*args, **kwargs))
                    elif fn == "test":
                        results.put(ob.test(*args, **kwargs))
                    elif fn == "test_loss":
                        results.put(ob.test_loss(*args, **kwargs))
                    else:
                        results.put(fn(*args, **kwargs))
            tasks.task_done()

    def start(self):
        self.process = mp.Process(
            target=ModelWorker.run,
            args=(self.model, self.tasks, self.results)
        )
        self.process.start()
        print(f"Started worker {self.process.pid} for model {self.model.name_or_path}")
        return self
    
    def stop(self):
        self.tasks.put(None)
        if self.process is not None:
            self.process.join()
        torch.cuda.empty_cache()
        return self

    def __call__(self, ob, fn, *args, **kwargs):
        self.tasks.put((deepcopy(ob), fn, args, kwargs))
        return self

def get_worker(params, eval=False):

    if ('gpt' not in params.model_path) and ('gemini' not in params.model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            params.tokenizer_path,
            trust_remote_code=True,
            **params.tokenizer_kwarg
        )
        if 'oasst-sft-6-llama-30b' in params.tokenizer_path:
            tokenizer.bos_token_id = 1
            tokenizer.unk_token_id = 0
        if 'guanaco' in params.tokenizer_path:
            tokenizer.eos_token_id = 2
            tokenizer.unk_token_id = 0
        if 'llama-2' in params.tokenizer_path or 'Llama-2' in params.tokenizer_path:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'falcon' in params.tokenizer_path:
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loaded tokenizer")

        raw_conv_template = get_conversation_template(params.conversation_template)

        if raw_conv_template.name == 'zero_shot':
            raw_conv_template.roles = tuple(['### ' + r for r in raw_conv_template.roles])
            raw_conv_template.sep = '\n'
        elif raw_conv_template.name == 'llama-2':
            raw_conv_template.sep2 = raw_conv_template.sep2.strip()

        conv_template = raw_conv_template
        print(f"Loaded conversation template")
        worker = ModelWorker(
                params.model_path,
                params.model_kwarg,
                tokenizer,
                conv_template,
                params.device
            )

        if not eval:
            worker.start()

        print('Loaded target LLM model')
    
    else:
        tokenizer = [None]
        worker = ModelWorker(
                params.model_path,
                None,
                None,
                None,
                None
            )
    return worker