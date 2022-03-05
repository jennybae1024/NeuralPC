# Dataset for NeuralPC

import os, sys, json, pickle, random
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from util.data_utils import pad_ids

BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"
UNK = "<unk>"
USER = "<speaker1>"
SYS = "<speaker2>"
CONT = "<context>"
NeuralPC_SPECIAL_TOKENS = {
    "bos_token": BOS,
    "eos_token": EOS,
    "pad_token": PAD,
    "unk_token": UNK,
    "additional_special_tokens": [USER, SYS, CONT],
}

class NeuralPCDataset(Dataset):
    def __init__(self, args, split, tokenizer=None):
        super().__init__()
        self.args = args
        self.split = split
        if not tokenizer:
            self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
            self.tok.add_special_tokens(NeuralPC_SPECIAL_TOKENS)
        else:
            self.tok = tokenizer
        self.pad_id = self.tok.pad_token_id
        self.load_data()

    def load_data(self):
        self.data = pd.read_csv(os.path.join(self.args.data_dir, 'neuralpc',
                                        f"{self.args.datafile_name}_{self.split}.csv"), index_col=0).to_dict('records')

        self.examples = []
        for sample in tqdm(self.data):
            inputs = self.tok(sample["goal"])
            targets = self.tok(sample["dialog"])

            self.examples.append({
                "input_ids": inputs["input_ids"],
                "target_ids": targets["input_ids"],
                "topic_flow": sample["topic_flow"],
                "did": sample["id"]
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        target_ids = [b["target_ids"] for b in batch]
        input_ids = torch.LongTensor(pad_ids(input_ids, self.pad_id))
        target_ids = torch.LongTensor(pad_ids(target_ids, -100))
        input_mask = input_ids.ne(self.pad_id).float()
        return input_ids, input_mask, target_ids


class NeuralPCTestDataset(NeuralPCDataset):
    def load_data(self):
        self.data = pd.read_csv(os.path.join(self.args.data_dir, 'neuralpc',
                                        f"{self.args.datafile_name}.csv"), index_col=0).to_dict('records')

        self.examples = []
        for sample in tqdm(self.data):
            inputs = self.tok(sample["goal"])
            targets = self.tok(sample["dialog"])

            self.examples.append({
                "input_ids": inputs["input_ids"],
                "target_ids": targets["input_ids"],
                "topic_flow": sample["topic_flow"],
                "did": sample["id"]
            })

GPT2NeuralPC_SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<sep>",
    "additional_special_tokens": [USER, SYS],
}

class NeuralPCDataset4LM(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        self.load_data()

    def load_data(self):
        raw_df = pd.read_csv(os.path.join(self.args.data_dir,
                                          f"{self.args.datafile_name}_{self.split}.csv"), index_col=0)
        self.text_from = raw_df["goal"].to_list()
        self.text_to = raw_df["dialog"].to_list()
        assert len(self.text_from) == len(self.text_to)

        if self.split != "test":
            for i in range(1):
                print(f"Sample {self.split} data ({i}) of goal instruction: {self.text_from[i]}")
                print(f"Sample {self.split} data ({i}) of dialog: {self.text_to[i]}")

    def __len__(self):
        return len(self.text_to)

    def __getitem__(self, idx):
        return self.text_from[idx], self.text_to[idx]