"""
Activation inversion for feature extraction and other datasets.

Two execution paths depending on --method:
  patchscopes  Extract hidden state from target model via nnsight, inject into
               reconstruction model (EmbedLlama/EmbedMistral + MRec wrapper).
               If a vec2text trainer checkpoint is provided via
               --reconstruction_model_path, uses the trainer directly and skips
               the manual nnsight extraction.
  lit          Use the LIT activation-patching pipeline (inversion_interpret).
"""

import json
import numpy as np
import os
import random
import torch

from dataclasses import dataclass
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.inversion_utils import (
    get_dataset,
    inversion_interpret,
    get_model,
    get_tokenizer,
)
from src.utils.model_utils import MRec
from src.utils.infra_utils import (
    get_tokenizer as get_tokenizer_lqa,
    get_model as get_model_lqa,
    update_config,
)

from inversion.vec2text import analyze_utils


@dataclass
class inversion_config:
    # Target model (activations are read from this model)
    activation_model_name: str = "MODEL_NAME"

    # Reconstruction model
    reconstruction_model_name: str = "MODEL_NAME"
    reconstruction_model_path: str = ""

    seed: int = 42
    batch_size: int = 16
    modify_chat_template: bool = True
    truncate: str = "none"
    output_dir: str = "results/"

    method: str = "lit"
    layer_idx: int = 15
    split: str = "valid"
    dataset_name: str = "feature_extraction"
    max_new_tokens: int = 20

    # Feature extraction / generic dataset args
    data_dir: str = ""
    task_name: str = ""
    process_all_tasks: bool = False
    n_samples: int = -1


def main(**kwargs):
    args = inversion_config()
    update_config(args, **kwargs)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    trainer = None
    act_tokenizer = None
    reconstruct_tokenizer = None
    act_model = None
    reconstruct_model = None

    if args.method == "patchscopes":
        # Try loading a vec2text trainer checkpoint first. If that fails (e.g.
        # checkpoint was saved in a layout analyze_utils doesn't recognise),
        # fall back to loading the models manually with nnsight + MRec.
        try:
            _, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
                args.reconstruction_model_path,
            )
            # embeddings_from_layer_n is not saved in the checkpoint — must be set manually
            trainer.model.embeddings_from_layer_n = args.layer_idx
        except Exception:
            act_tokenizer = get_tokenizer(args.activation_model_name)
            reconstruct_tokenizer = get_tokenizer(args.reconstruction_model_name, reconstruct=True)

            act_model = get_model(
                args.activation_model_name,
                act_tokenizer,
                device="cuda",
                extract_act=True,
            )
            reconstruct_model = get_model(
                args.reconstruction_model_name,
                reconstruct_tokenizer,
                load_peft_checkpoint=args.reconstruction_model_path,
                device="cuda",
                reconstruct=True,
            )
            reconstruct_model = MRec(reconstruct_model, reconstruct_tokenizer)
    else:
        # lit path
        act_tokenizer = get_tokenizer_lqa(args.activation_model_name)
        reconstruct_tokenizer = get_tokenizer_lqa(args.reconstruction_model_name)
        act_model = get_model_lqa(args.activation_model_name, act_tokenizer, device="cuda")
        reconstruct_model = get_model_lqa(
            args.reconstruction_model_name,
            reconstruct_tokenizer,
            load_peft_checkpoint=args.reconstruction_model_path,
            device="cuda",
        )

    # Determine which tasks to process
    if args.process_all_tasks and args.dataset_name == "feature_extraction":
        task_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".tsv")])
        print(f"Processing all tasks: {task_files}")
    else:
        task_files = [args.task_name]

    for task_file in task_files:
        print(f"\n{'='*80}\nProcessing task: {task_file}\n{'='*80}\n")

        original_task_name = args.task_name
        args.task_name = task_file
        dataset = get_dataset(args, act_tokenizer, reconstruct_tokenizer, train=False)
        args.task_name = original_task_name

        dataset_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        outputs = {}
        for idx, item in enumerate(tqdm(dataset_dataloader, total=len(dataset_dataloader), desc="Inverting!")):
            read_prompts = item["read_prompt"]
            questions = item["dialog"][1]["content"]
            answers = item["dialog"][2]["content"]

            if args.method == "patchscopes" and trainer is not None:
                # vec2text trainer path: re-embeds the text internally
                tokenized_text = trainer.embedder_tokenizer(
                    read_prompts, return_tensors="pt", padding=True, truncation=True
                )
                tokenized_text["embedder_input_ids"] = tokenized_text["input_ids"]
                tokenized_text["embedder_attention_mask"] = tokenized_text["attention_mask"]
                sentences = trainer.tokenizer.batch_decode(
                    trainer.generate(
                        tokenized_text,
                        generation_kwargs={
                            "do_sample": False,
                            "max_new_tokens": args.max_new_tokens,
                        },
                    ),
                    skip_special_tokens=True,
                )
                for i in range(len(sentences)):
                    outputs[i + idx * args.batch_size] = {
                        "prompt": read_prompts[i],
                        "completion": sentences[i],
                        "question": questions[i],
                        "answer": answers[i],
                    }

            elif args.method == "patchscopes" and trainer is None:
                # Manual nnsight extraction + MRec injection
                with act_model.trace(read_prompts) as tracer:
                    if hasattr(act_model, "transformer"):
                        clean_hs = act_model.transformer.h[args.layer_idx].output[0].save()
                    else:
                        clean_hs = act_model.model.layers[args.layer_idx].output[0].save()

                activations = clean_hs[:, -1, :]
                input_ids = torch.full(
                    (activations.size()[0], 1),
                    reconstruct_model.act_token_id,
                ).to(activations.device)
                attention_mask = torch.ones(input_ids.shape).to(activations.device)
                model_outputs = reconstruct_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    activations=activations,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                cleaned_outputs = model_outputs[:, input_ids.size()[1]:]
                sentences = reconstruct_tokenizer.batch_decode(cleaned_outputs, skip_special_tokens=True)
                for i in range(len(sentences)):
                    outputs[i + idx * args.batch_size] = {
                        "prompt": read_prompts[i],
                        "completion": sentences[i],
                        "question": questions[i],
                        "answer": answers[i],
                    }

            else:
                # lit path
                read_prompts_nested = [[read_prompts[i]] for i in range(len(read_prompts))]
                questions_nested = [[questions[i]] for i in range(len(questions))]
                _, outputs_gen, batch = inversion_interpret(
                    act_model,
                    reconstruct_model,
                    act_tokenizer,
                    reconstruct_tokenizer,
                    read_prompts_nested,
                    questions_nested,
                    generate=True,
                    print_output=False,
                    args=args,
                )
                for i in range(len(outputs_gen)):
                    outputs[i + idx * args.batch_size] = {
                        "prompt": act_tokenizer.decode(
                            batch["tokenized_read"]["input_ids"][i], skip_special_tokens=True
                        ),
                        "completion": outputs_gen[i],
                        "question": questions[i],
                        "answer": answers[i],
                    }

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        name = task_file.split(".")[0].replace(" ", "_")
        out_path = f"{args.output_dir}/{name}_{args.split}.json"
        with open(out_path, "w") as f:
            json.dump(outputs, f, indent=2)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    Fire(main)
