"""
Infra utils
"""

import logging
import json
import os
import torch
from copy import deepcopy
from collections import OrderedDict
from dataclasses import is_dataclass
from datetime import datetime
from functools import partial
from glob import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    _or_policy,
    lambda_auto_wrap_policy,
)
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

# Chat templates for encoding (with reasoning tokens for some models)
ENCODER_CHAT_TEMPLATES = {
    "meta-llama/Llama-3.1-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    "mistralai/Mistral-Small-24B-Instruct-2501": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
}

# Chat templates for decoding
DECODER_CHAT_TEMPLATES = {
    "mistralai/Mistral-Small-24B-Instruct-2501": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "mistralai/Ministral-8B-Instruct-2410": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "mistralai/Mistral-7B-Instruct-v0.3": "{{- bos_token }}\n\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}",
    "meta-llama/Meta-Llama-3-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Llama-3.1-8B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "meta-llama/Meta-Llama-3-70B-Instruct": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set role = message['role'] %}{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<|reserved_special_token_8|><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<|reserved_special_token_8|>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<|reserved_special_token_8|><think>\\n'}}{% endif %}",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<|reserved_special_token_8|><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<|reserved_special_token_8|>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<|reserved_special_token_8|><think>\\n'}}{% endif %}",
}


# PAD token IDs for different models
PAD_TOKEN_IDS = {
    "meta-llama/Llama-3.1-8B": 128001,
    "meta-llama/Meta-Llama-3-8B-Instruct": 128010,
    "meta-llama/Llama-3.1-8B-Instruct": 128010,
    "meta-llama/Meta-Llama-3-70B-Instruct": 128010,
    "mistralai/Mistral-7B-v0.3": 2,
    "mistralai/Ministral-8B-Instruct-2410": 999,
    "mistralai/Mistral-Small-24B-Instruct-2501": 2,
    "mistralai/Mistral-7B-Instruct-v0.3": 2,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 128010,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 151643,
}


def update_config(config, **kwargs):
    """
    Update a dataclass config object with keyword arguments.
    Supports nested dataclass updates via dot notation (e.g., "parent.child").
    """
    def update_nested(obj, key, value):
        if hasattr(obj, key):
            if is_dataclass(getattr(obj, key)):
                update_config(
                    getattr(obj, key),
                    **{k: v for k, v in kwargs.items() if k.startswith(f"{key}.")},
                )
            else:
                setattr(obj, key, value)
        elif "." in key:
            parent, child = key.split(".", 1)
            if hasattr(obj, parent):
                update_nested(getattr(obj, parent), child, value)
        else:
            print(f"Warning: {type(obj).__name__} does not accept parameter: {key}")

    for k, v in kwargs.items():
        update_nested(config, k, v)


def get_tokenizer(model_name):
    """
    Load a tokenizer for the given model with appropriate padding settings.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", add_eos_token=True
    )
    tokenizer.pad_token_id = PAD_TOKEN_IDS.get(model_name, tokenizer.pad_token_id)

    # Handle special tokens for specific models
    if "distill-qwen" in model_name.lower():
        tokenizer.add_tokens(["<|reserved_special_token_8|>"])

    return tokenizer


def get_model(
    model_name,
    tokenizer,
    peft_config=None,
    load_peft_checkpoint=None,
    load_model_checkpoint=None,
    fsdp_args=None,
    device=None,
    rank=None,
    distributed_training=False,
):
    from peft import get_peft_model, PeftModel
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
    )
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from functools import partial

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
        if load_model_checkpoint not in (None, ""):
            model = AutoModelForCausalLM.from_pretrained(
                load_model_checkpoint,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                use_cache=None,
                device_map="auto" if device == "auto" else None,
            )
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

    assert peft_config is None or load_peft_checkpoint is None
    use_peft = peft_config is not None or load_peft_checkpoint is not None
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
    elif load_peft_checkpoint is not None:
        model = PeftModel.from_pretrained(model, load_peft_checkpoint)
        if distributed_training:
            for name, param in model.named_parameters():
                if "lora" in name or "adapter" in name:
                    param.requires_grad = True

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


def clean_text(text):
    """
    Clean generated text by removing chat template artifacts and extracting
    the prompt and completion.
    """
    text = text.split("[INST]")[-1]
    if "[/INST]" in text:
        prompt, completion = text.split("[/INST]")
        return prompt.strip().replace("</s>", ""), completion.strip().replace("</s>", "")
    if "<｜User｜>" in text:
        text = text.split("<｜User｜>")[-1]
        if "<|reserved_special_token_8|>" in text:
            prompt, completion = text.split("<|reserved_special_token_8|>")
            return (
                prompt.strip(),
                completion.split("<think>")[1].replace("<｜end▁of▁sentence｜>", "").strip(),
            )
        else:
            prompt, completion = text.split("<｜User｜>")
            return (
                prompt.strip(),
                completion.split("<think>")[1].replace("<｜end▁of▁sentence｜>", "").strip(),
            )
    if "Sure, I've analyzed the assistant." in text:
        text = text.split(
            "Sure, I've analyzed the assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>"
        )[1]
    prompt, completion = text.split("<|eot_id|>", 1)
    if "assistant<|end_header_id|>" in completion:
        completion = (
            text.split("assistant<|end_header_id|>")[1]
            .replace("<|end_of_text|>", "")
            .replace("<|eot_id|>", "")
        )
    elif "reflect<|end_header_id|>" in completion:
        completion = (
            text.split("reflect<|end_header_id|>")[1]
            .replace("<|end_of_text|>", "")
            .replace("<|eot_id|>", "")
        )
    return prompt.split("\n\n")[-1].strip(), completion.strip()


def get_modules(
    target_model,
    decoder_model,
    min_layer_to_read=15,
    max_layer_to_read=16,
    num_layers_to_read=1,
    layer_to_write=0,
    module_setup="read-vary_write-fixed_n-fixed",
    **kwargs,
):
    """
    Get the modules to read from and write to for activation patching.

    Args:
        target_model: Model to read activations from
        decoder_model: Model to write activations to
        min_layer_to_read: Minimum layer index to read
        max_layer_to_read: Maximum layer index to read
        num_layers_to_read: Number of layers to read
        layer_to_write: Layer index to write to
        module_setup: Configuration for module selection

    Returns:
        module_read: List of lists of modules to read from
        module_write: List of lists of modules to write to
    """
    # Determine the model structure
    try:
        eval("target_model.model.layers")
        target_model_str = "target_model.model"
    except:
        try:
            eval("target_model.model.model.layers")
            target_model_str = "target_model.model.model"
        except:
            target_model_str = "target_model.module.model.model"
    try:
        eval("decoder_model.model.layers")
        decoder_model_str = "decoder_model.model"
    except:
        try:
            eval("decoder_model.model.model.layers")
            decoder_model_str = "decoder_model.model.model"
        except:
            decoder_model_str = "decoder_model.module.model.model"

    # List[List[Module]]
    module_read, module_write = [], []
    for i in range(min_layer_to_read, max_layer_to_read):
        module_read_i, module_write_i = [], []
        if module_setup == "read-vary_write-vary_n-fixed":
            for j in range(i, i + num_layers_to_read):
                module_read_i.append(eval(f"{target_model_str}.layers[{j}]"))
                module_write_i.append(eval(f"{decoder_model_str}.layers[{j}]"))
        elif module_setup == "read-vary_write-vary_n-vary":
            for j in range(i):
                module_read_i.append(eval(f"{target_model_str}.layers[{j}]"))
                module_write_i.append(eval(f"{decoder_model_str}.layers[{j}]"))
        elif module_setup == "read-vary_write-fixed_n-fixed":
            for j in range(i, i + num_layers_to_read):
                module_read_i.append(eval(f"{target_model_str}.layers[{j}]"))
            for j in range(layer_to_write, layer_to_write + num_layers_to_read):
                module_write_i.append(eval(f"{decoder_model_str}.layers[{j}]"))
        else:
            raise NotImplementedError
        module_read.append(module_read_i)
        module_write.append(module_write_i)
    return module_read, module_write


##########################
###### Training utils ####
##########################


def create_logger(logging_dir, rank):
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.propagate = False

    if rank == 0:
        logger.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            fmt="[\033[34m%(asctime)s\033[0m] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_formatter = logging.Formatter(
            fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(f"{logging_dir}/log.txt")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.ERROR)
        logger.addHandler(logging.NullHandler())
    return logger


def get_logger(args, rank):
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.output_dir}/*"))
        experiment_dir = f"{args.output_dir}/{experiment_index:03d}"
        args.checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
        with open(f"{experiment_dir}/exp_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    else:
        logger = create_logger(None, rank)
    return logger


def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from src.configs.wandb_config import wandb_config as WANDB_CONFIG
    from dataclasses import asdict

    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = asdict(wandb_config)
    if train_config.run_name != "":
        init_dict["name"] = train_config.run_name
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    if fsdp_config is not None:
        run.config.update(fsdp_config)
    return run


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    if ema_model is None:
        return
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if param.requires_grad:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def get_ema(model, decay, device):
    ema = None
    if decay < 1:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
    update_ema(ema, model, decay=0)
    return ema


def save_model(decoder_model, ema_model, tokenizer, args, epoch, steps, logger, rank):
    """FSDP-aware checkpoint save (no optimizer/scheduler). Used by the latentqa path."""
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
        (output_dir, decoder_model, "model"),
        (ema_output_dir, ema_model, "ema model"),
    ]:
        if model is None:
            continue
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)
        if rank == 0:
            model.save_pretrained(dir, state_dict=state_dict)
            tokenizer.save_pretrained(dir)
            logger.info(f"{name} is saved in {dir} directory")


def fsdp_auto_wrap_policy(model, transformer_layer_name):
    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
        ),
    )
    auto_wrap_policy = partial(
        _or_policy, policies=[lambda_policy, transformer_wrap_policy]
    )
    return auto_wrap_policy
