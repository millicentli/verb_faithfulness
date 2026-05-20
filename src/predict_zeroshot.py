"""
Zero-shot prediction over feature extraction datasets.
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
    DataCollatorForPrediction,
    PredictionDataset,
    get_feature_extraction_datasets,
)
from src.utils.infra_utils import (
    get_model,
    get_tokenizer,
    update_config,
)


@dataclass
class zeroshot_config:
    # Model
    model_name: str = "MODEL_NAME"

    seed: int = 42
    batch_size: int = 16
    modify_chat_template: bool = True
    truncate: str = "none"
    save_name: str = ""
    output_dir: str = "results/"
    dataset_name: str = "dataset_name"

    # Feature extraction-specific args
    data_dir: str = ""
    task_name: str = ""
    n_samples: int = -1
    n_tokens: int = 20

    # Inversion output args: when True, load from a pre-generated inversion JSON
    # rather than the raw feature-extraction dataset. Expects files of the form
    # {data_dir}/{task_name}_{split}.json written by predict_inversion.py.
    from_file: bool = False
    split: str = "valid"


def main(
    **kwargs,
):
    args = zeroshot_config()
    update_config(args, **kwargs)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    tokenizer = get_tokenizer(args.model_name)
    model = get_model(args.model_name, tokenizer, device="cuda")

    if args.from_file:
        inversion_path = os.path.join(args.data_dir, f"{args.task_name}_{args.split}.json")
        eval_dataloader = DataLoader(
            PredictionDataset(tokenizer, inversion_path, args.dataset_name, from_file=True),
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=DataCollatorForPrediction(tokenizer),
        )
    elif args.dataset_name == "feature_extraction":
        eval_dataset = get_feature_extraction_datasets(args)
        eval_dataloader = DataLoader(
            PredictionDataset(tokenizer, eval_dataset[args.task_name], args.dataset_name),
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=DataCollatorForPrediction(tokenizer),
        )
    else:
        eval_dataloader = DataLoader(
            PredictionDataset(tokenizer, args.data_dir, args.dataset_name),
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=DataCollatorForPrediction(tokenizer),
        )

    results = {}
    print(f"\tNumber of samples for dataset {args.task_name}: {len(eval_dataloader)}")
    for idx, batch in tqdm(
        enumerate(eval_dataloader), total=len(eval_dataloader), desc="Running evaluation..."
    ):
        batch = {
            k: v.to("cuda") if k != "labels" and k != "original_prompts" else v \
            for k, v in batch.items()
        }

        outputs = model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_new_tokens=args.n_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        pruned_outputs = outputs[:, batch['input_ids'].shape[1]:]
        outputs_decoded = tokenizer.batch_decode(pruned_outputs, skip_special_tokens=True)
        prompt = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        for i, (out, p, l) in enumerate(zip(outputs_decoded, prompt, batch['labels'])):
            results[idx * args.batch_size + i] = {
                "prompt": p,
                "completion": out,
                "answer": l,
            }

    # Save the outputs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    name = args.task_name.split(".")[0]
    with open(f"{args.output_dir}/{name}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    Fire(main)