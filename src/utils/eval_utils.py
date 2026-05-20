"""
Dataset utils for the evals (reconstruction, prediction)
Some code adopted from LatentQA (Pan et al. 2024, https://arxiv.org/abs/2412.08686) with modifications.
"""

import json
import os

import pandas as pd
from torch.utils.data import Dataset

from src.utils.infra_utils import DECODER_CHAT_TEMPLATES, ENCODER_CHAT_TEMPLATES


class ModifiedLatentQADataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_path,
    ):
        self.tokenizer = tokenizer
        self.chat_template = DECODER_CHAT_TEMPLATES.get(tokenizer.name_or_path, None)
        self.dataset = json.load(open(f"{dataset_path}/outputs_valid.json", "r"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[f"{idx}"]

        completion = item["completion"]
        items_split = completion.split("None")

        qa_dialog = [
            {"role": "user", "content": item["questions"]},
            {"role": "assistant", "content": item["answer"]},
        ]

        if len(items_split) == 1:
            read_prompt = [
                {"role": "user", "content": items_split[0] + qa_dialog[0]['content']},
            ]
            read_prompt = self.tokenizer.apply_chat_template(
                read_prompt,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template,
            )
        elif len(items_split) == 3:
            read_prompt = [
                {"role": "user", "content": items_split[0]},
                {"role": "assistant", "content": items_split[1]},
                {"role": "user", "content": items_split[2] + qa_dialog[0]['content']},
            ]
            read_prompt = self.tokenizer.apply_chat_template(
                read_prompt,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template,
            )
        elif len(items_split) == 4:
            read_prompt = [
                {"role": "user", "content": items_split[0]},
                {"role": "assistant", "content": items_split[1]},
                {"role": "user", "content": items_split[2]},
                {"role": "assistant", "content": items_split[3]},
                {"role": "user", "content": qa_dialog[0]['content']},
            ]
            read_prompt = self.tokenizer.apply_chat_template(
                read_prompt,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template,
            )
        else:
            raise ValueError("Invalid number of splits in completion")

        return {
            "original_prompt": item['prompt'],
            "read_prompt": read_prompt,
            "label": qa_dialog[1]['content']
        }


class PredictionDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_name,
        from_file=False,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.from_file = from_file
        self.chat_template = ENCODER_CHAT_TEMPLATES.get(tokenizer.name_or_path, None)
        if self.dataset_name == "feature_extraction" and not from_file:
            self.dataset = dataset
        else:
            self.dataset = json.load(
                open(os.path.join(dataset), "r")
            )

        self.prefix = {
            "feature_extraction": "Complete the sentence: ",
            "personaqa": "",
        }[dataset_name]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset_name == "feature_extraction" and not self.from_file:
            items = self.dataset.iloc[idx]
            prompt = items['prompt_source']
            question = items['prompt_target']
            object = items['object']
        else:
            items = self.dataset[str(idx)]
            question = items['question']

            # Reconstruction case
            if 'completion' in items:
                prompt = items['completion']
            else:
                prompt = items['prompt']
            if 'target' in items:
                object = items['target']
            else:
                object = items['answer']

        if "llama" in self.tokenizer.name_or_path.lower():
            if self.dataset_name == "feature_extraction":
                read_prompt = [
                    {"role": "assistant", "content": prompt},
                    {"role": "user", "content": f"{self.prefix}{question.replace('x', '').strip()}"}
                ]
            elif self.dataset_name == "personaqa":
                read_prompt = [
                    {"role": "assistant", "content": prompt},
                    {"role": "user", "content": f"{self.prefix}{question.replace('x', '').strip()}"},
                ]
        else:
            if self.dataset_name == "feature_extraction":
                read_prompt = [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": prompt},
                    {"role": "user", "content": f"{self.prefix}{question.replace('x', '').strip()}"}
                ]
            else:
                read_prompt = [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": prompt},
                    {"role": "user", "content": f"{self.prefix}{question}"}
                ]

        read_prompt = self.tokenizer.apply_chat_template(
            read_prompt,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self.chat_template,
        )

        return {
            "original_prompt": prompt,
            "read_prompt": read_prompt,
            "label": object
        }


class DataCollatorForPrediction:
    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        text_tokenized = self.tokenizer(
            [b['read_prompt'] for b in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        return {
            "input_ids": text_tokenized["input_ids"],
            "attention_mask": text_tokenized["attention_mask"],
            "labels": [b['label'] for b in batch],
            "original_prompts": [b['original_prompt'] for b in batch],
        }


class VerbalizationDataset:
    def __init__(self, dataset, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset_name == "feature_extraction":
            items = self.dataset.iloc[idx]
            prompt_source = items['prompt_source']
            prompt_target = items['prompt_target']
            object = items['object']
        else:
            items = self.dataset[str(idx)]
            prompt_source = items['prompt']
            prompt_target = items['question']
            object = items['target']

        return {
            'prompt_source': prompt_source,
            'prompt_target': prompt_target,
            'object': object
        }


def get_feature_extraction_datasets(args):
    """
    Code is mostly taken from Patchscopes, with some modifications.
    """
    datasets_to_df = {}
    path = args.data_dir

    for filename in os.listdir(path):
        # Skip non-TSV files
        if not filename.endswith('.tsv'):
            continue

        df = pd.read_csv(f"{os.path.join(path, filename)}", sep="\t")
        df = df[df['prompt_source'].notna()]
        df = df[~df['prompt_source'].str.contains('\n')].reset_index(drop=True)

        if "star_constellation" in filename:
            df = df[~df["prompt_source"].str.contains("service")].reset_index(drop=True)
        elif "object_superclass" in filename:
            df = df[~df["prompt_source"].str.contains("Swainson ' s hawk and the prairie")].reset_index(drop=True)

        if args.n_samples > 0 and args.n_samples < len(df):
            df = df.sample(n=args.n_samples, replace=False, random_state=42).reset_index(drop=True)

        datasets_to_df[filename] = df

    return datasets_to_df