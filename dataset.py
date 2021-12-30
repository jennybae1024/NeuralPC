# Dataset for NeuralPC

import os, sys, json, pickle, random
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from util.data_utils import pad_ids

data_dir = '/media/disk1/jennybae/data/controllable_generation'

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
        if not self.args.data_dir:
            self.args.data_dir = data_dir
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