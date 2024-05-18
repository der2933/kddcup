import os
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from trl.trainer.utils import DataCollatorForCompletionOnlyLM
from accelerate import Accelerator
from peft import LoraConfig
import random
import re
random.seed(42)

os.environ["WANDB_PROJECT"] = "ma-rlhf"
os.environ["WANDB_RUN_NAME"] = "sft"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from utils import (
    ScriptArguments,
    DEFINE_EOS_TOKEN,
    DEFINE_PAD_TOKEN,
    formatting_alpaca_func,
    formatting_alpaca_func_bached,
    is_main_process,
    create_peft,
)

parser = HfArgumentParser(ScriptArguments)
train_args: ScriptArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

dataset_name = train_args.dataset_name
model_name = train_args.model_name
deepspeed_config_name = train_args.deepspeed_config_name
seq_length = train_args.seq_length
batch_size = train_args.batch_size
output_name = train_args.output_name
is_peft = train_args.use_QLora
use_flash_attention_2 = train_args.use_flash_attention_2
dataset_sub_name = None
num_train_epochs = train_args.num_train_epochs
gradient_accumulation_steps = train_args.gradient_accumulation_steps
learning_rate = train_args.learning_rate

def create_datasets(dataset_name, dataset_sub_name):
    dataset = load_dataset(dataset_name)
    return dataset, None


def create_model_tokenizer(name):
    # QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    device_map = {"": Accelerator().local_process_index}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config if not is_peft else None,
        device_map=device_map,
        # use_flash_attention_2=use_flash_attention_2, # gpt 2 not support flash attention2
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                trust_remote_code=True,
                                                padding_side='left',
                                                # model_max_length=1024
                                                )
    tokenizer.add_special_tokens({'pad_token': DEFINE_PAD_TOKEN})
    model.pad_token_id = tokenizer.pad_token_id
    model.pad_token = tokenizer.pad_token
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def create_sft_datasets(datasets, tokenizer, format_func, seq_length=512):
    train_dataset = datasets["train"]

    # train_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     train_dataset,
    #     formatting_func=format_func,
    #     infinite=True,
    #     seq_length=seq_length,
    #     shuffle=True,
    # )

    return train_dataset, None

def create_collator(tokenizer):
    '''
    ref https://github.com/huggingface/trl/blob/main/tests/test_data_collator_completion_only.py
    '''
    # instruction_template = "###Question: "
    response_template = "###Answer:"
    response_template_id = tokenizer.encode(
            response_template, add_special_tokens=False
        )[1:]
    return DataCollatorForCompletionOnlyLM(response_template_id, tokenizer=tokenizer)


def train():
    model, tokenizer = create_model_tokenizer(model_name)
    datasets, _ = create_datasets(dataset_name, dataset_sub_name)
    format_fun = formatting_alpaca_func_bached # for sft collarter
    train_datasets, _ = create_sft_datasets(datasets, tokenizer, format_fun, seq_length)
    collator = create_collator(tokenizer)

    # peft
    peft_config = create_peft(is_peft)

    training_args = TrainingArguments(
        output_dir=output_name,
        save_strategy='epoch',
        logging_steps=1,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed=deepspeed_config_name,
        report_to='wandb',
        lr_scheduler_type='cosine',
        # max_steps=10,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_datasets,
        max_seq_length=seq_length,
        peft_config=peft_config,
        packing=False,
        tokenizer=tokenizer,
        data_collator=collator,
        formatting_func=format_fun,
        dataset_num_proc=24,
    )
    trainer.train()
    trainer.save_model(output_name)


if __name__ == "__main__":
    train()
