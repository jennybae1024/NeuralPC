# Dataset for DialogNLI

import os, sys, json, pickle, random
from tqdm import tqdm
import pandas as pd
import jsonlines
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
)

dnli_dir = '/media/disk1/jennybae/data/dialogue_nli/'

class dnliDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.data = self._load_data()

        self.label2id = {"negative": 0,
                         "neutral": 1,
                         "positive": 2}
        self.num_labels = len(self.label2id)

    def _load_data(self):
        if self.args.data_dir:
            return jsonlines.Reader(open(os.path.join(self.args.data_dir, "dialogue_nli/dnli/dialogue_nli",
                                                      f'dialogue_nli_{self.split}.jsonl'))).read()

        else:
            return jsonlines.Reader(open(os.path.join(dnli_dir + "dnli/dialogue_nli",
                                               f'dialogue_nli_{self.split}.jsonl'))).read()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sent1 = self.tok.encode(sample["sentence1"])
        sent2 = self.tok.encode(sample["sentence2"])
        label = self.label2id[sample["label"]]

        input_ids = sent1 + sent2[1:]
        token_type_ids = [0] * len(sent1) + [1] * (len(sent2) - 1)
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(attention_mask) == len(token_type_ids)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label}

    def collate_fn(self, batch):
        input_ids = [torch.LongTensor(ele["input_ids"]) for ele in batch]
        # token_type_ids = [torch.LongTensor(ele["token_type_ids"]) for ele in batch]
        attention_mask = [torch.LongTensor(ele["attention_mask"]) for ele in batch]
        label = torch.tensor([ele["label"] for ele in batch])

        input_ids = pad_sequence(input_ids, True, padding_value=self.tok.pad_token_id).long()
        # token_type_ids = pad_sequence(token_type_ids, True, padding_value=0).long()
        attention_mask = pad_sequence(attention_mask, True, padding_value=0).long()

        return input_ids, attention_mask, label


job_rel = ['has_profession',
          'teach',
          'job_status',
          'previous_profession',
          'employed_by_company',
          'employed_by_general',
          'want_job']

class dnliExceptJob(dnliDataset):
    def _load_data(self):
        if self.args.data_dir:
            data_all = jsonlines.Reader(open(os.path.join(self.args.data_dir, "dialogue_nli/dnli/dialogue_nli",
                                                      f'dialogue_nli_{self.split}.jsonl'))).read()
        else:
            data_all = jsonlines.Reader(open(os.path.join(dnli_dir + "dnli/dialogue_nli",
                                               f'dialogue_nli_{self.split}.jsonl'))).read()

        data_out = []
        if self.split == "train":
            for ele in data_all:
                rel1 = ele["triple1"][1]
                rel2 = ele["triple2"][1]
                if rel1 not in job_rel and rel2 not in job_rel:
                    data_out.append(ele)
        else:
            data_out = data_all
        return data_out


class DnliMnliDataset(dnliDataset):
    def _load_data(self):
        if self.args.data_dir:
            nliData = jsonlines.Reader(open(os.path.join(self.args.data_dir, "dialogue_nli/dnli/dialogue_nli",
                                                          f'dialogue_nli_{self.split}.jsonl'))).read()
        else:
            nliData = jsonlines.Reader(open(os.path.join(dnli_dir + "dnli/dialogue_nli",
                                                          f'dialogue_nli_{self.split}.jsonl'))).read()

        if self.split == "train":
            convDict = {"entailment": "positive",
                        "contradiction": "negative",
                        "neutral": "neutral"}
            mnliData = []
            if self.args.data_dir:
                with open(os.path.join(self.args.data_dir, f'multinli_1.0/multinli_1.0_{self.split}.jsonl'), 'r') as json_file:
                    json_list = list(json_file)
            else:
                with open(f'/media/disk1/jennybae/data/multinli_1.0/multinli_1.0_{self.split}.jsonl', 'r') as json_file:
                    json_list = list(json_file)
            for json_str in json_list:
                ele = json.loads(json_str)
                mnliData.append({"sentence1": ele["sentence1"],
                                 "sentence2": ele["sentence2"],
                                 "label": convDict[ele["gold_label"]],
                                 "id": ele["pairID"]
                                 })
            nliData.extend(mnliData)
        return nliData


class DnliMnliExceptJob(DnliMnliDataset):
    def _load_data(self):
        if self.args.data_dir:
            data_all = jsonlines.Reader(open(os.path.join(self.args.data_dir, "dialogue_nli/dnli/dialogue_nli",
                                                      f'dialogue_nli_{self.split}.jsonl'))).read()
        else:
            data_all = jsonlines.Reader(open(os.path.join(dnli_dir + "dnli/dialogue_nli",
                                               f'dialogue_nli_{self.split}.jsonl'))).read()
        nliData = []
        if self.split == "train":
            for ele in data_all:
                rel1 = ele["triple1"][1]
                rel2 = ele["triple2"][1]
                if rel1 not in job_rel and rel2 not in job_rel:
                    nliData.append(ele)
        else:
            nliData = data_all

        if self.split == "train":
            convDict = {"entailment": "positive",
                        "contradiction": "negative",
                        "neutral": "neutral"}
            mnliData = []
            if self.args.data_dir:
                with open(os.path.join(self.args.data_dir, f'multinli_1.0/multinli_1.0_{self.split}.jsonl'), 'r') as json_file:
                    json_list = list(json_file)
            else:
                with open(f'/media/disk1/jennybae/data/multinli_1.0/multinli_1.0_{self.split}.jsonl', 'r') as json_file:
                    json_list = list(json_file)
            for json_str in json_list:
                ele = json.loads(json_str)
                mnliData.append({"sentence1": ele["sentence1"],
                                 "sentence2": ele["sentence2"],
                                 "label": convDict[ele["gold_label"]],
                                 "id": ele["pairID"]
                                 })
            nliData.extend(mnliData)
        return nliData




