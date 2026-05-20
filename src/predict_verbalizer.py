"""
Using two different types of verbalization techniques for evaluation
(1) Patchscopes (untrained + single activation)
(2) Latent Interpretation Tuning (finetuned + multiple activations)
"""

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.eval_utils import (
    VerbalizationDataset,
    get_feature_extraction_datasets,
)
from src.utils.infra_utils import (
    clean_text,
    get_model,
    get_tokenizer,
    update_config,
)
from src.utils.patchscopes_utils import (
    patchscopes,
    setup,
)
from src.utils.reading_utils import interpret


@dataclass
class verbalizer_predict_config:
    # Model
    target_model_name: str = "MODEL_NAME"
    decoder_model_name: str = "MODEL_NAME"
    decoder_model_path: str = ""

    num_tgt_layers: int = 16
    min_layer_to_read: int = 15
    max_layer_to_read: int = 16
    num_layers_to_read: int = 1
    num_layers_to_sample: int = 1
    layer_to_write: int = 0
    module_setup: str = "read-vary_write-fixed_n-fixed"

    seed: int = 42
    batch_size: int = 16
    modify_chat_template: bool = True
    truncate: str = "none"
    save_name: str = ""
    output_dir: str = "results/"
    split_gpus: bool = False
    method: str = "lit"
    dataset_name: str = "dataset_name"

    # Feature extraction-specific args
    data_dir: str = ""
    task_name: str = ""
    n_samples: int = -1
    n_tokens: int = 20
    src_layer: int = 15
    tgt_layer: int = 0
    min_tgt_layer: int = -1  # If set, will use range(min_tgt_layer, max_tgt_layer)
    max_tgt_layer: int = -1  # If set, will use range(min_tgt_layer, max_tgt_layer)
    process_all_tasks: bool = False  # If True, process all .tsv files in data_dir


def main(
    **kwargs,
):
    args = verbalizer_predict_config()
    update_config(args, **kwargs)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models once (outside task loop for efficiency)
    if args.method == "lit":
        act_tokenizer = get_tokenizer(args.target_model_name)
        decoder_tokenizer = get_tokenizer(args.decoder_model_name)

        if args.split_gpus:
            target_model = get_model(args.target_model_name, act_tokenizer, device="cuda:0")
            decoder_model = get_model(
                args.decoder_model_name,
                decoder_tokenizer,
                load_peft_checkpoint=args.decoder_model_path,
                device="cuda:1",
            )
        else:
            target_model = get_model(args.target_model_name, act_tokenizer, device="cuda")

            decoder_model = get_model(
                args.decoder_model_name,
                decoder_tokenizer,
                load_peft_checkpoint=args.decoder_model_path,
                device="cuda",
            )
    else:
        act_tokenizer, target_model, decoder_tokenizer, decoder_model = setup(args)

        target_model.cuda()
        decoder_model.cuda()

    # Determine which tasks to process
    if args.process_all_tasks and args.dataset_name == "feature_extraction":
        # Get all .tsv files from data_dir
        task_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.tsv')])
        print(f"Processing all tasks: {task_files}")
    else:
        # Single task mode
        task_files = [args.task_name]

    # Process each task
    for task_file in task_files:
        print(f"\n{'='*80}")
        print(f"Processing task: {task_file}")
        print(f"{'='*80}\n")

        # Load dataset for this task
        if args.dataset_name == "feature_extraction":
            # Temporarily set task_name for dataset loading
            original_task_name = args.task_name
            args.task_name = task_file
            eval_dataset = get_feature_extraction_datasets(args)
            args.task_name = original_task_name

            eval_dataloader = DataLoader(
                VerbalizationDataset(eval_dataset[task_file], args.dataset_name),
                batch_size=args.batch_size,
                pin_memory=True,
                shuffle=False,
                collate_fn=lambda x: x,
            )
        else:
            with open(os.path.join(args.data_dir, f"{task_file}.json")) as f:
                eval_dataset = json.load(f)
            eval_dataloader = DataLoader(
                VerbalizationDataset(eval_dataset, args.dataset_name),
                batch_size=args.batch_size,
                pin_memory=True,
                shuffle=False,
                collate_fn=lambda x: x,
            )


        # Prepare output directories and files for multi-layer patchscopes
        layer_output_files = {}
        results = {}
        if args.method == "patchscopes" and (args.min_tgt_layer >= 0 and args.max_tgt_layer >= 0):
            name = task_file.split(".")[0].replace(" ", "_")
            tgt_layers = list(range(args.min_tgt_layer, args.max_tgt_layer))
            for tgt_layer in tgt_layers:
                layer_output_dir = f"{args.output_dir.replace(f'src_{args.src_layer}', f'src={args.src_layer}_tgt={tgt_layer}')}"
                if not os.path.exists(layer_output_dir):
                    os.makedirs(layer_output_dir)
                layer_output_files[tgt_layer] = f"{layer_output_dir}/{name}.json"
                results[tgt_layer] = {}

        print(f"\tNumber of samples for dataset {task_file}: {len(eval_dataloader)}")
        for i, batch in tqdm(
            enumerate(eval_dataloader), total=len(eval_dataloader), desc="Running evaluation..."
        ):
            prompt_source_batch = np.array([b['prompt_source'] for b in batch])
            prompt_source_batch = [[prompt] for prompt in prompt_source_batch]
            object_batch = np.array([b['object'] for b in batch])

            if args.dataset_name == "personaqa":
                prompt_target = np.array([[b['prompt_target']] for b in batch])
            else:
                prompt_target = np.array([[batch[0]['prompt_target']]])

            if args.method == "lit":
                if args.dataset_name != "personaqa":
                    prompt_target = np.array([[prompt_target[0][0].replace("x", "").strip()]])
                elif args.dataset_name == "personaqa":
                    prompt_target = np.array([[prompt[0].replace("x", "").strip()] for prompt in prompt_target])

                _, out, _ = interpret(
                    target_model,
                    decoder_model,
                    act_tokenizer,
                    decoder_tokenizer,
                    prompt_source_batch,
                    prompt_target,
                    generate=True,
                    print_output=False,
                    args=args
                )
                for idx, key, target in zip(range(len(out)), prompt_source_batch, object_batch):
                    _, completion = clean_text(decoder_tokenizer.decode(out[idx]))
                    completion_tokenized = decoder_tokenizer.encode(completion)
                    completion_truncated = completion_tokenized[:args.n_tokens]
                    final_completion = decoder_tokenizer.decode(completion_truncated)
                    results[i * args.batch_size + idx] = {
                        "prompt": key[0],
                        "completion": final_completion,
                        "answer": target,
                    }
            else:
                # Duplicate the tgts since they're the same
                if args.dataset_name == "personaqa":
                    new_targets = prompt_target
                else:
                    new_targets = [prompt_target[0] for _ in range(len(prompt_source_batch))]

                # Determine target layers: single or multiple
                if args.min_tgt_layer >= 0 and args.max_tgt_layer >= 0:
                    tgt_layers = list(range(args.min_tgt_layer, args.max_tgt_layer))
                else:
                    tgt_layers = args.tgt_layer

                patchscopes_result = patchscopes(
                    prompt_source_batch,
                    new_targets,
                    target_model,
                    decoder_tokenizer,
                    decoder_model,
                    src_layer=args.src_layer,
                    tgt_layer=tgt_layers,
                    n_tokens=args.n_tokens,
                    print_output=False,
                )

                # Handle both single and multiple target layers
                if isinstance(tgt_layers, int):
                    # Single layer: unpack tuple
                    _, out, _ = patchscopes_result
                    results_to_save = {tgt_layers: (out, prompt_source_batch, object_batch)}
                else:
                    # Multiple layers: results is a dict
                    results_to_save = {}
                    for layer, (_, out, _) in patchscopes_result.items():
                        results_to_save[layer] = (out, prompt_source_batch, object_batch)

                # Process results for each target layer
                for tgt_layer_key, (out, src_batch, obj_batch) in results_to_save.items():
                    if tgt_layer_key not in results:
                        results[tgt_layer_key] = {}

                    for idx, key, target in zip(range(len(out)), src_batch, obj_batch):
                        decoded_output = decoder_tokenizer.decode(out[idx], skip_special_tokens=True)

                        results[tgt_layer_key][i * args.batch_size + idx] = {
                            "prompt": key[0],
                            "completion": decoded_output,
                            "answer": target,
                        }

                    # Save results immediately for this layer (incremental saving)
                    if args.method == "patchscopes" and (args.min_tgt_layer >= 0 and args.max_tgt_layer >= 0):
                        with open(layer_output_files[tgt_layer_key], "w") as f:
                            json.dump(results[tgt_layer_key], f, indent=2)

        #  Save the outputs (only for non-patchscopes or single-layer patchscopes)
        if not (args.method == "patchscopes" and (args.min_tgt_layer >= 0 and args.max_tgt_layer >= 0)):
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)

            name = task_file.split(".")[0]
            # Replace spaces in the name with "_"
            name = name.replace(" ", "_")

            with open(f"{args.output_dir}/{name}.json", "w") as f:
                json.dump(results, f, indent=2)

            print(f"Results saved to: {args.output_dir}/{name}.json")
        else:
            # Extract base directory for the print message
            base_dir = args.output_dir.rsplit('/src_', 1)[0] if '/src_' in args.output_dir else args.output_dir.rstrip('/')
            print(f"Results saved incrementally for each target layer in {base_dir}/src={args.src_layer}_tgt=*/")


if __name__ == "__main__":
    Fire(main)