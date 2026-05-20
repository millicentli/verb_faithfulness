"""
Dataset utilities for tokenization and formatting
Simplified version extracted from latentqa
"""

import torch

from src.utils.infra_utils import DECODER_CHAT_TEMPLATES

# Constants
IGNORE_IDX = -100

MODEL_DIRS = "/scratch/li.mil/verb_faithfulness/personaqa"

# Magic numbers for token shifts
NUM_READ_TOKENS_TO_SHIFT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 1,
    "meta-llama/Llama-3.1-8B-Instruct": 1,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1,
    "mistralai/Ministral-8B-Instruct-2410": 2,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "mistralai/Mistral-7B-Instruct-v0.3": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 2,
    f"{MODEL_DIRS}/Llama-3-8B-PersonaQA": 1, # Change this with yours
    f"{MODEL_DIRS}/Llama-3-8B-PersonaQA-Shuffled": 1, # Change this with yours
    f"{MODEL_DIRS}/Llama-3-8B-PersonaQA-Fantasy": 1, # Change this with yours
}

NUM_WRITE_TOKENS_TO_SHIFT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 5,
    "meta-llama/Llama-3.1-8B-Instruct": 5,
    "meta-llama/Meta-Llama-3-70B-Instruct": 5,
    "mistralai/Ministral-8B-Instruct-2410": 2,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "mistralai/Mistral-7B-Instruct-v0.3": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 2,
    f"{MODEL_DIRS}/Llama-3-8B-PersonaQA": 5, # Change this with yours
    f"{MODEL_DIRS}/Llama-3-8B-PersonaQA-Shuffled": 5, # Change this with yours
    f"{MODEL_DIRS}/Llama-3-8B-PersonaQA-Fantasy": 5, # Change this with yours
}

# Magic numbers that correspond to the token idxs of the chat format for the models
CHAT_FORMAT_TOKENS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "meta-llama/Meta-Llama-3-70B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
        torch.tensor([128006, 36013, 128007, 271]),
    ),
    "mistralai/Ministral-8B-Instruct-2410": (
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([4]),
    ),
    "mistralai/Mistral-Small-24B-Instruct-2501": (
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([4]),
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": (
        torch.tensor([3]),
        torch.tensor([4]),
        torch.tensor([4]),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": (
        torch.tensor([128011]),
        torch.tensor([128012]),
        torch.tensor([128016]),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": (
        torch.tensor([151644]),
        torch.tensor([151645]),
        torch.tensor([151665]),
    ),
}

# Dialog format
BASE_DIALOG = [
    {
        "role": "assistant",
        "content": "Sure, I've analyzed the assistant.",
    }
]


def mask_inputs(
    input_ids,
    tokenizer_name,
    get_verb_mask=None,
    shift_start=False,
    mask_all_but_last=False,
    modify_chat_template=False,
    inversion=False,
):
    """Mask inputs for activation extraction."""
    start_tokens, end_tokens_default, end_tokens_modify = CHAT_FORMAT_TOKENS[
        tokenizer_name
    ]
    end_tokens = end_tokens_modify if modify_chat_template else end_tokens_default
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(batch_size):
        start_idx = []
        end_idx = []
        for i in range(seq_len):
            if torch.equal(input_ids[b][i : i + len(start_tokens)], start_tokens):
                start_idx.append(i)
            if torch.equal(input_ids[b][i : i + len(end_tokens)], end_tokens):
                end_idx.append(i)

        # Mask based on the mode
        if inversion:
            mask[b][: end_idx[-1] + len(end_tokens)] = True
            continue
        elif get_verb_mask == "user":
            if len(start_idx) == 1:
                continue
            mask[b][start_idx[0] : start_idx[1]] = True
        elif get_verb_mask == "system":
            mask[b][1 : start_idx[0]] = True
        else:
            assert get_verb_mask is None
            if len(start_idx) != len(end_idx):
                mask[b][:] = True
                continue

            if mask_all_but_last:
                mask[b][: end_idx[-1] + len(end_tokens)] = True
            else:
                for i, (start, end) in enumerate(zip(start_idx, end_idx)):
                    if shift_start and i == 0:
                        mask[b][start - 1 : end + len(end_tokens)] = True
                    else:
                        mask[b][start : end + len(end_tokens)] = True
    return mask


def tokenize(
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
    """Tokenize batch for activation patching."""
    read_name = act_tokenizer.name_or_path if read_name is None else read_name
    write_name = decoder_tokenizer.name_or_path if write_name is None else write_name

    # Tokenize read inputs
    tokenized_read = act_tokenizer(
        [item["read_prompt"] for item in batch],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenized_batch = {"tokenized_read": tokenized_read}

    # Compute length of read input
    read_lengths = torch.sum(tokenized_read.attention_mask, dim=1)
    tokenized_batch["read_lengths"] = read_lengths - 1  # Exclude BOS token

    if get_verb_mask is not None:
        verb_mask = mask_inputs(
            tokenized_read.input_ids, read_name, get_verb_mask=get_verb_mask
        )
        verb_lengths = torch.sum(verb_mask, dim=1)
        pad_lengths = read_lengths - verb_lengths
        tokenized_batch["verb_lengths"] = verb_lengths
    else:
        pad_lengths = read_lengths

    # Tokenize dialog inputs
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

    # Compute length of write input
    write_lengths = torch.sum(tokenized_write.attention_mask, dim=1)
    tokenized_batch["write_lengths"] = write_lengths - NUM_WRITE_TOKENS_TO_SHIFT[write_name]

    # Add labels for training
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
        )
        assert decoder_tokenizer.padding_side == "left"
        tokenized_write["labels"] = tokenized_write.input_ids.clone()
        mask = (tokenized_write.attention_mask == 0) | user_inputs_mask
        tokenized_write["labels"][mask] = IGNORE_IDX
    return tokenized_batch
