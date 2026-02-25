from src.model.reward_model import Qwen3VLRewardModel
from src.model.utils import find_target_linear_names
from src.data.collator import Qwen3VLDataCollator
from src.data.dataset import convert_GSB_csv_to_reward_data

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor
from trl import get_kbit_device_map, get_quantization_config
from datasets import load_dataset, DatasetDict

def build_reward_model(
    model_config,
    peft_lora_config,
    training_args,
):
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    elif training_args.fp32:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float32
        
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=True if training_args.gradient_checkpointing else False,
    )

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = Qwen3VLRewardModel.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        **model_kwargs
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(torch_dtype)
    
    # create lora and peft model
    if peft_lora_config.lora_enable:
        target_modules = find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude
        )
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    return model, processor, peft_config

def build_data_collator(processor):
    collator = Qwen3VLDataCollator(processor)
    return collator

def build_dataset(data_config):
    dataset: DatasetDict = load_dataset('csv', data_files=data_config.meta_file)
    
    def add_idx(example, idx):
        example['metainfo_idx'] = idx
        return example
    
    dataset['train'] = dataset['train'].map(
        lambda example, idx: add_idx(example, idx), 
        with_indices=True
    ) 
    
    if not data_config.use_tied_data:
        filter_func = lambda example: any(example[f"{dim}"] != "same" for dim in data_config.eval_dim)
        dataset = dataset.filter(filter_func)

    # convert data to reward data
    convert_func = lambda example: convert_GSB_csv_to_reward_data(
        example, 
        data_config.data_dir, 
        data_config.eval_dim, 
        data_config.max_frame_pixels, 
        data_config.fps, 
        data_config.num_frames,
        data_config.prompt_template_type,
        sample_type=data_config.sample_type,
    )
    dataset = dataset.map(
        convert_func, 
        remove_columns=dataset['train'].column_names, 
        load_from_cache_file=False
    )
    return dataset['train']