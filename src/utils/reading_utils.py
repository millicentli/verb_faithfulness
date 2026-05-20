"""
Reading utils for LIT.
Code adopted from LatentQA (Pan et al. 2024, https://arxiv.org/abs/2412.08686).
"""

import numpy as np
import torch

from src.utils.activation_utils import latent_qa
from src.utils.dataset_utils import BASE_DIALOG, tokenize
from src.utils.infra_utils import clean_text, get_modules, ENCODER_CHAT_TEMPLATES


def interpret(
    target_model,
    decoder_model,
    act_tokenizer,
    decoder_tokenizer,
    dialogs,
    questions,
    args,
    generate=True,
    print_output=True,
    replacement_list=None,
):
    """
    Interpret activations by reading from target_model and writing to decoder_model.

    Args:
        target_model: Model to read activations from
        decoder_model: Model to write activations to
        act_tokenizer: Tokenizer for activation model
        decoder_tokenizer: Tokenizer for decoder model
        dialogs: List of dialog prompts
        questions: List of questions to ask
        args: Configuration arguments
        generate: Whether to generate text
        print_output: Whether to print outputs
        replacement_list: Optional list of replacements

    Returns:
        QA_PAIRS: Dictionary mapping dialogs to QA pairs
        out: Model outputs
        batch: Tokenized batch
    """
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)

    module_read, module_write = get_modules(target_model, decoder_model, **vars(args))
    chat_template = ENCODER_CHAT_TEMPLATES.get(act_tokenizer.name_or_path, None)

    # Validate dialogs
    if all([len(d) == 1 for d in dialogs]):
        assert args.truncate == "none"
    elif min([len(d) for d in dialogs]) == max([len(d) for d in dialogs]):
        pass
    else:
        assert False

    probe_data = []
    for idx, dialog in enumerate(dialogs):
        if len(dialog) == 1:
            if chat_template != None:
                read_prompt = act_tokenizer.apply_chat_template(
                    [{"role": "user", "content": dialog[0]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=chat_template,
                )
            else:
                read_prompt = dialog[0]
        elif len(dialog) == 2:
            read_prompt = act_tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": dialog[0]},
                    {"role": "assistant", "content": dialog[1]},
                ],
                tokenize=False,
                chat_template=chat_template,
            )
        else:
            read_prompt = act_tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": dialog[0]},
                    {"role": "assistant", "content": dialog[1]},
                    {"role": "user", "content": dialog[2]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template,
            )

        if len(questions) == len(dialogs):
            if generate:
                dialog = [{"role": "user", "content": questions[idx][0]}]
            else:
                dialog = [
                    {"role": "user", "content": questions[idx][0]},
                    {"role": "assistant", "content": questions[idx][1]},
                ]
            probe_data.append(
                {
                    "read_prompt": read_prompt,
                    "dialog": BASE_DIALOG + dialog,
                }
            )
        else:
            for i, item in enumerate(questions):
                question = item[0]
                if replacement_list is not None:
                    if replacement_list[idx * len(questions) + i] is not None:
                        question = question.replace("<BLANK>", replacement_list[idx * len(questions) + i])
                if generate:
                    dialog = [{"role": "user", "content": question}]
                else:
                    dialog = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": item[1]},
                    ]
                probe_data.append(
                    {
                        "read_prompt": read_prompt,
                        "dialog": BASE_DIALOG + dialog,
                    }
                )

    batch = tokenize(
        probe_data,
        act_tokenizer,
        decoder_tokenizer,
        read_name=args.target_model_name,
        write_name=args.decoder_model_name,
        generate=generate,
        get_verb_mask=args.truncate if args.truncate != "none" else None,
        modify_chat_template=args.modify_chat_template,
    )

    out = latent_qa(
        batch,
        target_model,
        decoder_model,
        module_read[0],
        module_write[0],
        act_tokenizer,
        decoder_tokenizer,
        shift_position_ids=False,
        generate=generate,
        cache_target_model_grad=False,
    )

    QA_PAIRS = {}
    if generate:
        for i in range(len(out)):
            if i % len(questions) == 0:
                curr_dialog = dialogs[i // len(questions)][0]
                QA_PAIRS[curr_dialog] = []
            prompt, completion = clean_text(decoder_tokenizer.decode(out[i]))
            if print_output:
                print(f"[PROMPT]: {questions[i % len(questions)]}")
                print(f"[COMPLETION]: {completion}")
                print("#" * 80)
            QA_PAIRS[curr_dialog].append((prompt, completion))

    return QA_PAIRS, out, batch
