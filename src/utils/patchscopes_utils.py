"""
Utils for Patchscopes, including Patchscopes implementation.
"""

from typing import List

import nnsight
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.infra_utils import PAD_TOKEN_IDS


def load_model(model_name, device=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=None,
        device_map=device,
    )

    return model


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", add_eos_token=True
    )
    tokenizer.pad_token_id = PAD_TOKEN_IDS.get(model_name, tokenizer.pad_token_id)
    return tokenizer


def setup(args):
    act_tokenizer = load_tokenizer(args.target_model_name)
    decoder_tokenizer = load_tokenizer(args.decoder_model_name)

    if args.split_gpus:
        target_model = load_model(args.target_model_name, device="cuda:0")
        decoder_model = load_model(args.decoder_model_name, device="cuda:1")
    else:
        target_model = load_model(args.target_model_name, device="cuda")
        decoder_model = load_model(args.decoder_model_name, device="cuda")

    # Wrap the model with nnsight
    target_model = nnsight.LanguageModel(target_model, tokenizer=act_tokenizer)
    decoder_model = nnsight.LanguageModel(decoder_model, tokenizer=decoder_tokenizer)
    return act_tokenizer, target_model, decoder_tokenizer, decoder_model


def patchscopes(
    dialogs,
    questions,
    target_model,
    decoder_tokenizer,
    decoder_model,
    src_layer=15,
    tgt_layer=0,
    n_tokens=100,
    print_output=True,
    generate=True,
):
    """
    Patchscopes: Patch activations from source to target model.

    Args:
        tgt_layer: Can be either an int (single layer) or a list of ints (multiple layers)

    Returns:
        If tgt_layer is an int: (QA_PAIRS, out, prompt_source_batch)
        If tgt_layer is a list: dict mapping tgt_layer -> (QA_PAIRS, out, prompt_source_batch)
    """
    prompt_source_batch = dialogs
    prompt_target = questions

    if isinstance(prompt_source_batch[0], List):
        prompt_source_batch = [item[0] for item in prompt_source_batch]
        prompt_target = [item[0] for item in prompt_target]

    # Get the representation of the last token in the prompt source batch (only once!)
    with target_model.trace(prompt_source_batch) as tracer:
        if hasattr(target_model, 'model'):
            clean_hs = target_model.model.layers[src_layer].output[0][:, -1, :].save()
        else:
            clean_hs = target_model.transformer.h[src_layer].output[0][:, -1, :].save()

    # Handle both single layer and multiple layers
    tgt_layers = [tgt_layer] if isinstance(tgt_layer, int) else tgt_layer
    results_by_layer = {}

    for current_tgt_layer in tgt_layers:
        # Transplant this into the last token to get the generation
        with decoder_model.generate(max_new_tokens=n_tokens) as tracer:
            with tracer.invoke(prompt_target) as invoker:
                if hasattr(decoder_model, 'model'):
                    decoder_model.model.layers[current_tgt_layer].output[0][:, -1, :] = clean_hs
                else:
                    decoder_model.transformer.h[current_tgt_layer].output[0][:, -1, :] = clean_hs

            # Now generate
            out = decoder_model.generator.output.save()

        QA_PAIRS = {}
        if generate:
            for i in range(len(out)):
                curr_dialog = prompt_source_batch[i]
                if curr_dialog not in QA_PAIRS:
                    QA_PAIRS[curr_dialog] = []
                prompt = prompt_source_batch[i]
                completion = decoder_tokenizer.decode(out[i])
                try:
                    completion = completion.split(decoder_tokenizer.bos_token)[1]
                except:
                    completion = completion

                # Remove the question
                cleaned_completion = completion.replace(questions[i][0], "")

                # Tokenize and remove special tokens
                cleaned_completion = decoder_tokenizer.decode(
                    decoder_tokenizer(cleaned_completion)['input_ids'], skip_special_tokens=True
                )
                if print_output:
                    print(f"[SRC LAYER {src_layer} -> TGT LAYER {current_tgt_layer}]")
                    print(f"[PROMPT]: {questions[i]}")
                    print(f"[COMPLETION]: {cleaned_completion}")
                    print("#" * 80)
                QA_PAIRS[curr_dialog].append((prompt, cleaned_completion))

        results_by_layer[current_tgt_layer] = (QA_PAIRS, out, prompt_source_batch)

    # Return format: single layer returns tuple, multiple layers returns dict
    if isinstance(tgt_layer, int):
        return results_by_layer[tgt_layer]
    else:
        return results_by_layer