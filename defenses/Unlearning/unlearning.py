import argparse
import logging
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model, PeftModel
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers import DataCollatorForLanguageModeling
from utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
    ensure_folder,
)

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)
def create_rlhf_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=1):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        fraction: <1 will do downsampling.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["original_prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            #adv_prompt = examples["adv_prompt"][i]
            prompt = examples["original_prompt"][i] + examples["adv_prompt"][i]
            response_list = []
            response_list.append(examples["language_model_output"][i])

            # Add only bad samples.

            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"### Question: {prompt}\n ### Answer: {response}"
                tokenized = tokenizer(text, truncation=True, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                print(len(test_tokenized["input_ids"]) - 1)
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(
        preproccess,
        batched=True,
        remove_columns=[
            "original_prompt",
            "per_prompt",
            "target",
            "adv_prompt",
            "language_model_output",
            "attack_iterations",
            "data_id",
            "is_JB"
        ],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def main(args) -> None:
    # print args
    # print(args)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    # If use LoRA.
    if args.use_lora:
        print("Using LoRA")
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token

    # Load harmful data.
    train_dataset = load_dataset('json', data_files="../data/Unlearning/unlearning_data.json", split="train")

    train_bad_loader = create_rlhf_dataloader_from_dataset(
        tokenizer, train_dataset, batch_size=args.batch_size
    )


    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    (
        model,
        optimizer,
        train_bad_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_bad_loader, lr_scheduler
    )

    model.train()


    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    # Stop if bad loss is big enough or reaching max step.
    while bad_loss < args.max_bad_loss and idx < args.max_unlearn_steps:
        step = 0
        for bad_batch in train_bad_loader:
            ############ GA on answer only. ############
            bad_loss = get_answer_loss("ga", bad_batch, model, tokenizer)
            loss = args.bad_weight * bad_loss
            #Backprop.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step += 1
            # Print.
            stats = (
                f"batch: {step}, "
                f"bad_loss: {-bad_loss:.2f}, "
            )
            logging.info(stats)
            print(stats)
        idx += 1

        # Save model.
        if idx % args.save_every == 0:
            #ensure_folder(args.model_save_dir)
            model.save_pretrained(args.model_save_dir, from_pt=True)

    if args.use_lora:
        model = model.merge_and_unload(progressbar=True)

    # Save final model.

    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    merge_model = PeftModel.from_pretrained(pretrained_model, args.model_save_dir)
    merge_model = merge_model.merge_and_unload(progressbar=True)

    # Reload tokenizer to save it
    merge_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    data_name = args.data_dir.split("/")[-1].split(".")[0]
    model_name = args.model_name.split("/")[-1]
    merge_dir = f"./merged_models/unlearn_{model_name}_{data_name}_epochs{args.max_unlearn_steps}"
    merge_model.save_pretrained(merge_dir)
    merge_tokenizer.save_pretrained(merge_dir)
    logging.info("Unlearning finished")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", type=bool, default=True)

    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=2,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Unlearning LR.")
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="meta-llama/Llama-2-13b-chat-hf",
        default="lmsys/vicuna-13b-v1.3",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="defenses/data/Unlearning/unlearning_data_sample.json",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/vicuna-13b-v1.5_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="How many steps to save model."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/Llama-2_unlearned_0409.log",
        # help="Log file name",
    )
    parser.add_argument(
        "--device_id",
        type = int,
        default = '0',
        help = "device id"
    )
    parser.add_argument(
        "--ref_device_id",
        type = int,
        default = '1',
        help = "device id"
    )
    args = parser.parse_args()
    main(args)
