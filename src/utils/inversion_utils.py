"""
Inversion utilities for both inference and training.
Adapted from latentqa/lit/utils/inversion_utils.py for use in this repo.
"""

import json
import nnsight
import numpy as np
import os
import random
import re
import torch
import torch.distributed as dist

from datetime import datetime
from functools import partial
from glob import glob
from itertools import islice
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict

from src.utils.activation_utils import latent_qa
from src.utils.dataset_utils import (
    NUM_READ_TOKENS_TO_SHIFT,
    NUM_WRITE_TOKENS_TO_SHIFT,
    DECODER_CHAT_TEMPLATES,
    BASE_DIALOG,
    IGNORE_IDX,
    mask_inputs,
)
from src.utils.infra_utils import PAD_TOKEN_IDS
from src.utils.eval_utils import get_feature_extraction_datasets
from src.utils.infra_utils import get_modules, fsdp_auto_wrap_policy
from src.utils.model_utils import EmbedLlamaForCausalLM, EmbedMistralForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from peft import get_peft_model, PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


class InversionMSMARCODataset(Dataset):
    def __init__(self, act_tokenizer, decoder_tokenizer, dataset):
        self.act_tokenizer = act_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.lengths = []

        self.dataset = dataset.filter(lambda x: x['text'] != "")
        self.text = self.dataset['text']
        for idx in range(len(self.dataset)):
            self.lengths.append(len(self.text[idx]))

        self.dataset_length = len(self.dataset)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        read_prompt = self.text[idx]
        return {"read_prompt": read_prompt, "dialog": ""}


class InversionMillionPromptsDataset(Dataset):
    def __init__(self, act_tokenizer, decoder_tokenizer, dataset):
        self.act_tokenizer = act_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.lengths = []

        self.dataset = dataset.filter(lambda x: x['user'] != "")
        self.text = self.dataset['user']
        for idx in range(len(self.dataset)):
            self.lengths.append(len(self.text[idx]))

        self.dataset_length = len(self.dataset)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        read_prompt = self.text[idx]
        return {"read_prompt": read_prompt, "dialog": ""}


class InversionFeatureExtractionDataset(Dataset):
    def __init__(self, act_tokenizer, decoder_tokenizer, dataset):
        self.act_tokenizer = act_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        read_prompt = self.dataset.iloc[idx]["prompt_source"]
        qa_dialog = [
            {"role": "user", "content": self.dataset.iloc[idx]["prompt_target"]},
            {"role": "assistant", "content": self.dataset.iloc[idx]["object"]},
        ]
        return {"read_prompt": read_prompt, "dialog": BASE_DIALOG + qa_dialog}


class GenericInversionDataset(Dataset):
    def __init__(self, act_tokenizer, decoder_tokenizer, dataset):
        self.act_tokenizer = act_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[str(idx)]
        read_prompt = item["prompt"]
        qa_dialog = [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["target"]},
        ]
        return {"read_prompt": read_prompt, "dialog": BASE_DIALOG + qa_dialog}


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True) -> None:
        self.lengths = data_source.lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[: len(ids) // self.batch_size * self.batch_size]
        batches = [ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int,
                 shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas


def get_dist_batch_sampler(dataset, train_config, mode):
    return DistributedLengthBasedBatchSampler(
        dataset,
        train_config.batch_size_training,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=(mode == "train"),
        seed=train_config.seed,
    )


class DataCollatorForInversion:
    def __init__(self, act_tokenizer, decoder_tokenizer, get_verb_mask, generate=False,
                 mask_all_but_last=False, nudge_persona=False, modify_chat_template=False,
                 method="latentqa"):
        self.act_tokenizer = act_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        assert get_verb_mask in ["user", "system", None]
        self.get_verb_mask = get_verb_mask
        self.generate = generate
        self.mask_all_but_last = mask_all_but_last
        self.nudge = "Base your answers on my instructions. " if nudge_persona else ""
        self.modify_chat_template = modify_chat_template
        self.method = method

    def __call__(self, batch):
        formatted_batch = []
        if self.method == "patchscopes":
            for item in batch:
                formatted_batch.append(
                    {
                        "read_prompt": item["read_prompt"],
                        "dialog": [{"role": "assistant", "content": "<act>" + item["read_prompt"]}],
                        "labels": "<act>" + item["read_prompt"] + self.decoder_tokenizer.eos_token,
                    }
                )
            return tokenize_patchscopes(formatted_batch, self.decoder_tokenizer)
        else:
            for item in batch:
                formatted_batch.append(
                    {
                        "read_prompt": item["read_prompt"],
                        "dialog": [{"role": "assistant", "content": item["read_prompt"]}],
                        "label": item["read_prompt"],
                    }
                )
            return tokenize_latentqa(
                formatted_batch,
                self.act_tokenizer,
                self.decoder_tokenizer,
                get_verb_mask=None,
                generate=self.generate,
                mask_all_but_last=self.mask_all_but_last,
                modify_chat_template=self.modify_chat_template,
            )


def split_million_prompts_dataset(dataset):
    ds_train_valid = dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({"train": ds_train_valid["train"], "valid": ds_train_valid["test"]})


def get_dataloaders(train_config, act_tokenizer, decoder_tokenizer):
    if train_config.dataset_name == "latentqa":
        raise NotImplementedError("latentqa dataset not supported in this repo; use MSMARCO")
    elif train_config.dataset_name == "wentingzhao/one-million-instructions":
        dataset = load_dataset(train_config.dataset_name)
        split_dataset = split_million_prompts_dataset(dataset["train"])
        dataset_train = InversionMillionPromptsDataset(act_tokenizer, decoder_tokenizer, split_dataset["train"])
    else:
        dataset = load_dataset(train_config.dataset_name)
        split_dataset = dataset["train"].train_test_split(test_size=0.01)
        dataset_train = InversionMSMARCODataset(act_tokenizer, decoder_tokenizer, split_dataset["train"])

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=0,
        pin_memory=True,
        collate_fn=DataCollatorForInversion(
            act_tokenizer,
            decoder_tokenizer,
            get_verb_mask=None,
            mask_all_but_last=False,
            nudge_persona=train_config.nudge_persona,
            modify_chat_template=train_config.modify_chat_template,
            method=train_config.method,
        ),
        batch_sampler=get_dist_batch_sampler(dataset_train, train_config, "train"),
    )
    if train_config.eval_ppl:
        if train_config.dataset_name == "wentingzhao/one-million-instructions":
            dataset_eval = InversionMillionPromptsDataset(
                act_tokenizer, decoder_tokenizer, split_dataset["valid"]
            )
        else:
            dataset_eval = InversionMSMARCODataset(
                act_tokenizer, decoder_tokenizer, split_dataset["test"]
            )
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            num_workers=0,
            pin_memory=True,
            collate_fn=DataCollatorForInversion(
                act_tokenizer,
                decoder_tokenizer,
                get_verb_mask=None,
                mask_all_but_last=False,
                nudge_persona=train_config.nudge_persona,
                modify_chat_template=train_config.modify_chat_template,
                method=train_config.method,
            ),
            batch_sampler=get_dist_batch_sampler(dataset_eval, train_config, "val"),
        )
        return train_dataloader, eval_dataloader
    return train_dataloader, None


def get_dataset(args, act_tokenizer, decoder_tokenizer, train=False):
    if args.dataset_name == "feature_extraction":
        assert args.task_name != "" and args.data_dir != ""
        dataset = get_feature_extraction_datasets(args)
        return InversionFeatureExtractionDataset(
            act_tokenizer,
            decoder_tokenizer,
            dataset[args.task_name],
        )
    else:
        assert args.dataset_name in [
            "introspection",
            "next_token_prediction",
            "personaqa",
        ] and args.data_dir != ""
        dataset = json.load(open(args.data_dir, "r"))
        return GenericInversionDataset(act_tokenizer, decoder_tokenizer, dataset)


def tokenize_patchscopes(batch, decoder_tokenizer):
    tokenized = decoder_tokenizer(
        [item["labels"] for item in batch],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    act_token_id = decoder_tokenizer("<act>")["input_ids"][1]
    labels = tokenized.input_ids.clone()
    labels[labels == decoder_tokenizer.pad_token_id] = IGNORE_IDX
    labels[labels == act_token_id] = IGNORE_IDX
    input_texts = [item["read_prompt"] for item in batch]
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "input_text": input_texts,
        "labels": labels,
    }


def tokenize_latentqa(
    batch,
    act_tokenizer,
    decoder_tokenizer,
    read_name=None,
    write_name=None,
    generate=False,
    get_verb_mask=None,
    mask_all_but_last=False,
    modify_chat_template=False,
):
    read_name = act_tokenizer.name_or_path if read_name is None else read_name
    write_name = decoder_tokenizer.name_or_path if write_name is None else write_name

    tokenized_read = act_tokenizer(
        [item["read_prompt"] for item in batch],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenized_batch = {"tokenized_read": tokenized_read}
    read_lengths = torch.sum(tokenized_read.attention_mask, dim=1)
    tokenized_batch["read_lengths"] = read_lengths - 1

    if get_verb_mask is not None:
        verb_mask = mask_inputs(
            tokenized_read.input_ids, read_name, get_verb_mask=get_verb_mask
        )
        verb_lengths = torch.sum(verb_mask, dim=1)
        pad_lengths = read_lengths - verb_lengths
        tokenized_batch["verb_lengths"] = verb_lengths
    else:
        pad_lengths = read_lengths

    queries = []
    for i in range(len(pad_lengths)):
        query = [
            {
                "role": "user",
                "content": "? " * (pad_lengths[i] - NUM_READ_TOKENS_TO_SHIFT[read_name]),
            }
        ]
        query += batch[i]["dialog"]
        queries.append(
            decoder_tokenizer.apply_chat_template(
                query,
                tokenize=False,
                add_generation_prompt=generate,
                chat_template=(
                    DECODER_CHAT_TEMPLATES[write_name] if modify_chat_template else None
                ),
            )
        )
    tokenized_write = decoder_tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenized_batch["tokenized_write"] = tokenized_write

    write_lengths = torch.sum(tokenized_write.attention_mask, dim=1)
    tokenized_batch["write_lengths"] = write_lengths - NUM_WRITE_TOKENS_TO_SHIFT[write_name]

    if not generate:
        user_inputs_mask = mask_inputs(
            tokenized_write.input_ids,
            write_name,
            get_verb_mask=None,
            shift_start=any(
                [
                    m in write_name.lower()
                    for m in ["mistral", "llama-3", "deepseek-r1-distill"]
                ]
            ),
            mask_all_but_last=mask_all_but_last,
            modify_chat_template=modify_chat_template,
            inversion=True,
        )
        assert decoder_tokenizer.padding_side == "left"
        tokenized_write["labels"] = tokenized_write.input_ids.clone()
        mask = (tokenized_write.attention_mask == 0) | user_inputs_mask
        tokenized_write["labels"][mask] = IGNORE_IDX
    return tokenized_batch


def inversion_interpret(
    act_model,
    reconstruct_model,
    act_tokenizer,
    reconstruct_tokenizer,
    dialogs,
    questions,
    args,
    generate=True,
    print_output=False,
):
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    module_read, module_write = get_modules(act_model, reconstruct_model, **vars(args))

    probe_data = []
    for dialog, _ in zip(dialogs, questions):
        probe_data.append({"read_prompt": dialog[0], "dialog": ""})

    batch = tokenize_latentqa(
        probe_data,
        act_tokenizer,
        reconstruct_tokenizer,
        read_name=args.activation_model_name,
        write_name=args.reconstruction_model_name,
        generate=generate,
        get_verb_mask=args.truncate if args.truncate != "none" else None,
        modify_chat_template=args.modify_chat_template,
    )

    out = latent_qa(
        batch,
        act_model,
        reconstruct_model,
        module_read[0],
        module_write[0],
        act_tokenizer,
        reconstruct_tokenizer,
        shift_position_ids=False,
        generate=generate,
        cache_target_model_grad=False,
        max_new_tokens=256,
    )
    write_tokens = batch["tokenized_write"].input_ids
    outputs = out[:, write_tokens.shape[1]:]
    outputs_gen = reconstruct_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    QA_PAIRS = {}
    if generate:
        for i in range(len(outputs_gen)):
            curr_dialog = dialogs[i][0]
            if curr_dialog not in QA_PAIRS:
                QA_PAIRS[curr_dialog] = []
            prompt, completion = probe_data[i]["read_prompt"], outputs_gen[i]
            if print_output:
                print(f"[PROMPT]: {questions[i % len(questions)]}")
                print(f"[COMPLETION]: {completion}")
                print("#" * 80)
            QA_PAIRS[curr_dialog].append((prompt, completion))

    return QA_PAIRS, outputs_gen, batch


def get_tokenizer(model_name, reconstruct=False):
    """Load tokenizer, optionally adding the <act> special token for the MRec path.

    IMPORTANT: If reconstruct=True, the <act> token must be added BEFORE loading
    the model so that resize_token_embeddings is called with the correct vocab size.
    Forgetting this step causes silent garbage output during generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", add_eos_token=True
    )
    if reconstruct:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<act>"]})
    tokenizer.pad_token_id = PAD_TOKEN_IDS[model_name]
    if "distill-qwen" in model_name.lower():
        tokenizer.add_tokens(["<|reserved_special_token_8|>"])
    return tokenizer


def get_model(
    model_name,
    tokenizer,
    peft_config=None,
    load_peft_checkpoint=None,
    fsdp_args=None,
    device=None,
    rank=None,
    distributed_training=False,
    extract_act=False,
    reconstruct=False,
    args=None,
):
    """Load a causal LM for the inversion pipeline (supports inference and FSDP training).

    - extract_act=True: wraps in nnsight.LanguageModel for hidden-state extraction
    - reconstruct=True: loads EmbedLlama/EmbedMistral which accept an `activations` kwarg
    - fsdp_args: if set, wraps model in FSDP for distributed training
    """
    if fsdp_args is not None and fsdp_args.low_cpu_fsdp:
        from pkg_resources import packaging
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception(
                "latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                "please install latest nightly."
            )
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, use_cache=None, torch_dtype=torch.bfloat16,
            )
        else:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, use_cache=None, torch_dtype=torch.bfloat16,
                )
    else:
        if reconstruct:
            if "llama" in model_name.lower():
                model = EmbedLlamaForCausalLM.from_pretrained(
                    model_name,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    use_cache=None,
                    device_map="auto" if device == "auto" else None,
                )
            elif "ministral" in model_name.lower() or "mistral" in model_name.lower():
                model = EmbedMistralForCausalLM.from_pretrained(
                    model_name,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    use_cache=None,
                    device_map="auto" if device == "auto" else None,
                )
            else:
                raise ValueError(f"Model {model_name} not supported for reconstruction")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                use_cache=None,
                device_map="auto" if device == "auto" else None,
            )

    model.resize_token_embeddings(len(tokenizer))
    for _, param in model.named_parameters():
        param.requires_grad = False

    if extract_act:
        model = nnsight.LanguageModel(model, tokenizer=tokenizer).to(torch.bfloat16)

    # Load PEFT
    assert peft_config is None or load_peft_checkpoint is None
    use_peft = peft_config is not None or load_peft_checkpoint is not None
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
    elif load_peft_checkpoint is not None:
        # If the path doesn't contain "epoch", find the latest checkpoint automatically
        if "epoch" not in load_peft_checkpoint:
            experiment_index = glob(f"{load_peft_checkpoint}/*")
            latest_dir = max(experiment_index, key=os.path.getctime)
            num = int(latest_dir.split("/")[-1]) - 1
            second_to_latest_dir = "/".join(latest_dir.split("/")[:-1]) + f"/{num:03d}"
            pattern = f"{re.escape('steps')}(.*?){re.escape('-')}"
            all_files = glob(f"{second_to_latest_dir}/checkpoints/*")
            num_to_paths = {}
            for file in all_files:
                if "linear" in file:
                    continue
                filename = os.path.basename(file)
                match = re.search(pattern, filename)
                if match:
                    num = match.group(1)
                    num_to_paths[int(num)] = file
            try:
                max_key = max(num_to_paths.keys())
            except:
                raise ValueError("Trying to get the max of an empty directory! Delete it and try again.")
            load_peft_checkpoint = num_to_paths[max_key]
            if args is not None:
                args.load_model_checkpoint = load_peft_checkpoint

        model = PeftModel.from_pretrained(model, load_peft_checkpoint)
        if distributed_training:
            for name, param in model.named_parameters():
                if "lora" in name or "adapter" in name:
                    param.requires_grad = True

    # Distribute model
    if fsdp_args is None:
        if device is not None and device != "auto":
            model = model.to(device)
        if distributed_training:
            model = DDP(model, device_ids=[rank])
        return model
    else:
        hsdp_device_mesh = None
        if fsdp_args.hsdp and fsdp_args.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
            hsdp_device_mesh = hsdp_device_mesh(
                replica_group_size=fsdp_args.replica_group_size,
                sharding_group_size=fsdp_args.sharding_group_size,
            )

        if "llama" in model_name:
            DECODER_LAYER = LlamaDecoderLayer
        elif "mistral" in model_name:
            DECODER_LAYER = MistralDecoderLayer
        elif "qwen2" in model_name:
            DECODER_LAYER = Qwen2DecoderLayer

        wrapping_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DECODER_LAYER},
        )
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, DECODER_LAYER)
        device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if use_peft else wrapping_policy,
            cpu_offload=(CPUOffload(offload_params=True) if fsdp_args.fsdp_cpu_offload else None),
            mixed_precision=None,
            sharding_strategy=fsdp_args.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=fsdp_args.low_cpu_fsdp,
            param_init_fn=lambda module: (
                module.to_empty(device=torch.device("cuda"), recurse=False)
                if fsdp_args.low_cpu_fsdp and rank != 0
                else None
            ),
        )
        if fsdp_args.fsdp_activation_checkpointing:
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            check_fn = lambda submodule: isinstance(submodule, DECODER_LAYER)
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )
        return model


def save_model(decoder_model, ema_model, tokenizer, optimizer, scheduler, args, epoch, steps, logger, rank):
    """Inversion checkpoint save — includes optimizer and scheduler state."""
    if rank == 0:
        logger.info(f"Saving decoder model...")
        output_dir = (
            args.checkpoint_dir
            + f"/epoch{epoch}-steps{steps}"
            + f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        ema_output_dir = (
            args.checkpoint_dir
            + f"/ema-epoch{epoch}-steps{steps}"
            + f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    else:
        output_dir, ema_output_dir = None, None
    for dir, model, name in [
        (output_dir, decoder_model.model.module, "model"),
        (ema_output_dir, ema_model, "ema model"),
    ]:
        if model is None:
            continue
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)
        if rank == 0:
            model.save_pretrained(dir, state_dict=state_dict)
            tokenizer.save_pretrained(dir)
            torch.save(optimizer.state_dict(), f"{dir}/optimizer.pt")
            torch.save(scheduler.state_dict(), f"{dir}/scheduler.pt")
            logger.info(f"{name} is saved in {dir} directory")
