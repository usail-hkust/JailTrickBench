import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:

    """Forward pass through a LLM."""

    def __init__(
        self, 
        model,
        tokenizer,
        conv_template_name
    ):

        # Language model
        self.model = model

        # Tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        # if 'llama-2' in tokenizer_path:
        #     self.tokenizer.pad_token = self.tokenizer.unk_token
        # if not self.tokenizer.pad_token:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def __call__(self, batch, max_length=6000):

        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        # Forward pass through the LLM
        try:
            outputs = self.model.generate(
                batch_input_ids, 
                attention_mask=batch_attention_mask,
                max_length = max_length
            )
        except Exception as e:
            print(f"Error: {e}")
            return []

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]

        return batch_outputs