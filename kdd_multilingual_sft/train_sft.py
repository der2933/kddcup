# paged_adamw_8bit
import copy
import logging
import logging
import os
import numpy as np
import torch
import transformers

# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer
from typing import Dict, Optional, Sequence, List
from peft import LoraConfig, LoftQConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig, PeftMixedModel

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flashattn2: Optional[bool] = field(default=False, metadata={"help":"is using flash attention 2"})
    use_lora: Optional[bool] = field(default=False, metadata={"help":"PEFT: using lora to finetune"})
    lora_r: int  = field(default=16, metadata={"help":"lora rank"})
    lora_alpha: int  = field(default=32, metadata={"help":"lora alpha"})
    lora_dropout: float  = field(default=0.05, metadata={"help":"lora dropout rate"})
    use_qlora4bit: bool = field(default=False, metadata={"help":"using 4bit quantinize QLora to finetune"})
    do_eval_lora_ckp: str = field(default='', metadata={"help":"lora checkpoint to merge"})

@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={
        "help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={
        "help": "Path to the evaluation data."})
    source_length: int = field(default=2048)
    target_length: int = field(default=128)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_deepspeed: bool = field(default=False)




def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"
            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))
    # logger.info(extension)
    # logger.info(data_files)
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
        num_proc=10
    )['train']
    logger.info('load file done')
    return raw_datasets


IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a helpful and multilingual online shopping assistant. You can understand and respond to user queries in English, German, Italian, French, Japanese, Spanish, Chinese. You are knowledgeable about various products. NOTE:ONLY OUTPUT THE ANSWER!!\n\n"


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, source_length: int) -> Dict:
    """Tokenize a list of strings."""

    tokenized_list = [
        tokenizer.encode(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=source_length,
            add_special_tokens=False,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized[0]
                          for tokenized in tokenized_list]
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = [
        tokenized.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        mode: str
) -> Dict:
    """Preprocess the data by tokenizing."""

    if mode=='train':
        examples = [s + t + tokenizer.eos_token for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(
            strings, tokenizer, data_args.source_length) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    else:
        prompt_tokenied, labels_tokenized = [_tokenize_fn(strings, tokenizer, data_args.source_length) for strings in (sources, targets)]
        input_ids , labels = prompt_tokenied["input_ids"], labels_tokenized["input_ids"]
        
    return dict(input_ids=input_ids, labels=labels)


def make_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: str, data_args: DataArguments, mode: str) -> Dataset:
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(
        data_path=data_path,
    )
    logging.warning("Formatting inputs...")

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer, mode: str):
        ins_data = examples['instruction']
        if 'input' not in examples.keys():
            input_data = [""] * len(ins_data)
        else:
            input_data = examples['input']
        output = examples['output']
        len_ = len(ins_data)
        sources = [
            ins_data[i]+"\n"+input_data[i]
            if input_data[i] != "" else 
            ins_data[i]
                for i in range(len_)
        ]

        # sources = [i[-data_args.source_length:] if len(i)>data_args.source_length else i for i in sources]
        
        # using model chat template to format model input, not tokenize in this step
        sources = [
            tokenizer.apply_chat_template(
                conversation=[{"role":"system","content":SYSTEM_PROMPT}, {"role":"user","content":i}],
                add_generation_prompt=True,
                tokenize=False,
                truncation=True,
                max_length=data_args.source_length
            ) 
            for i in sources
        ]
        return preprocess(sources, output, tokenizer, data_args, mode)

    generate_sources_targets_p = partial(
        generate_sources_targets, tokenizer=tokenizer, mode=mode)


    dataset = dataset.map(
        function=generate_sources_targets_p,
        batched=True,
        desc="Running tokenizer on train dataset",
        num_proc=10
    )
    return dataset



def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments, data_args: DataArguments) -> tuple:
    load_model_args = dict(pretrained_model_name_or_path=model_args.model_name_or_path, 
                           cache_dir=training_args.cache_dir,
                           torch_dtype='auto',
                           trust_remote_code=True)
    if model_args.use_flashattn2:
        load_model_args["attn_implementation"] = "flash_attention_2"
    if not training_args.use_deepspeed:
        load_model_args["device_map"]='auto'
    if model_args.use_qlora4bit:
        quantization_config= BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        load_model_args["quantization_config"]=quantization_config
        
    model = AutoModelForCausalLM.from_pretrained(**load_model_args)

    if model_args.use_lora:
        if model_args.use_qlora4bit:
            model = prepare_model_for_kbit_training(model)

        logging.warning("Loading model to Lora")

        TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]

        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=TARGET_MODULES,
            lora_dropout=model_args.lora_dropout,
            # modules_to_save=['embed_tokens'],
            modules_to_save=['embed_tokens','lm_head'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)

        # logger.info('model.active_adapters:')
        # logger.info(model.active_adapters)
        # for name, para in model.named_parameters():
        #     if para.requires_grad==True:
        #         logger.info(name+'----requires grad')
        

    logger.info(model)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.truncation_side = "left" # truncation left ?
    model.config.use_cache = False
    return model, tokenizer

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing=True
    model, tokenizer = load_model_and_tokenizer(model_args, training_args, data_args)

    # check and set padding token
    if tokenizer.pad_token is None:
        # use eos_token as pad_token
        tokenizer.pad_token = tokenizer.eos_token
        # or add a new pad_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    with training_args.main_process_first(desc="loading and tokenization"):
        if training_args.do_train:
            train_dataset = make_dataset(
                tokenizer=tokenizer, data_path=data_args.train_data_path, data_args=data_args, mode='train')
            logger.info("example train dataset[0] formated:")
            logger.info(tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=False))
            logger.info('input_ids_len')
            logger.info(len(train_dataset[0]['input_ids']))

        if training_args.do_eval:
            eval_dataset = make_dataset(
                tokenizer=tokenizer, data_path=data_args.eval_data_path, data_args=data_args, mode='eval')
            logger.info("example train dataset[0] formated:")
            logger.info(tokenizer.decode(eval_dataset[0]['input_ids'], skip_special_tokens=False))

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=model,
                                           padding=True,
                                           label_pad_token_id=IGNORE_INDEX)

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset if training_args.do_train else None,
                      eval_dataset=eval_dataset if training_args.do_eval else None,
                      data_collator=data_collator)
    logger.info(trainer.args)

    if training_args.do_train:
    #     # logger.info(tokenizer.batch_decode([train_dataset[0]['input_ids']]))
        # for name, para in trainer.model.named_parameters():
            # if 'embed_tokens' in name or 'lm_head' in name:
            #     para.requires_grad=True
            # logger.info(name)
            # logger.info(para.grad)
            # if para.requires_grad==True:
            #     logger.info(name+'----requires grad')
        trainer.train()
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        logger.info('save train state and save trained model')

    if training_args.do_eval:
        from transformers.generation import GenerationConfig
        import gc
        # import re
        # pattern = re.compile(r'<start_of_turn>model(.*?)<end_of_turn>',re.S|re.M)

        with torch.no_grad():
            eval_dataloader = trainer.get_test_dataloader(eval_dataset)

            model = PeftModel.from_pretrained(model, model_args.do_eval_lora_ckp)


            print(model)
            print(model.active_adapters)
    
            model.eval()
            logger.info("####predict####")
            gc.collect()
            torch.cuda.empty_cache()
            
            predictions, references = [], []
            for batch in tqdm(eval_dataloader):
                input_ids, labels = batch['input_ids'].to(0), batch['labels'].to(0)
    
                # greedy search
                pred = model.generate(inputs=input_ids,
                                            generation_config=GenerationConfig(
                                                num_beams=1,
                                                # early_stopping=False,
                                                do_sample=False,
                                                # top_k=10,
                                                # top_p=0.9,
                                                # length_penalty=1.2,
                                                no_repeat_ngram_size=5,
                                                max_new_tokens=data_args.target_length,
                                                eos_token_id=tokenizer.eos_token_id,
                                                bos_token_id=tokenizer.bos_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                            ),
                )
                batch_input_ids_len = input_ids.shape[1]
                # print(tokenizer.batch_decode(pred,skip_special_tokens=True))
                pred = tokenizer.batch_decode([i[batch_input_ids_len:] for i in pred], skip_special_tokens=True)
                labels[labels==-100] = tokenizer.pad_token_id
                ref = tokenizer.batch_decode(labels, skip_special_tokens=True)

                try:
                    # predictions += [pattern.findall(i)[0].strip() for i in pred]
                    predictions += [i.strip() for i in pred]
                except Exception as e:
                    print(e)
                    print(pred)
                    predictions += pred
                references += ref
        import sacrebleu

        logger.info(sacrebleu.corpus_bleu(predictions, [references]))

        with open(training_args.output_dir+'/predicitons.txt','w') as pred_file:
            for i in predictions:
                pred_file.write(i.replace('\n', '')+'\n')
        with open(training_args.output_dir+'/references.txt','w') as ref_file:
            for i in references:
                ref_file.write(i+'\n')


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()