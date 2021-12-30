# Dataset for TopicNLI

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

topic_nli_dir = '/media/disk1/jennybae/data/topic_nli/'
topic_dict = {'education': ['attend_school', 'school_status', 'has_degree'],
             'favorite activity': ['favorite_activity', 'like_activity'],
             'favorite animal': ['favorite_animal', 'like_animal'],
             'favorite book': ['favorite_book', 'like_read'],
             'favorite color': ['favorite_color'],
             'favorite food': ['favorite_food', 'like_food', 'favorite_drink', 'like_drink'],
             'favorite movie': ['favorite_movie', 'like_movie'],
             'favorite music': ['favorite_music', 'like_music', 'favorite_music_artist'],
             'favorite place': ['favorite_place', 'like_goto'],
             'favorite season': ['favorite_season'],
             'favorite show': ['favorite_show', 'like_watching'],
             'favorite sport': ['favorite_sport', 'like_sports'],
             'hobby': ['favorite_hobby', 'has_hobby'],
             'ability': ['has_ability'],
             'job': ['has_profession','teach','job_status','previous_profession','employed_by_company',
                      'employed_by_general','want_job'],
             'family': ['have_chidren', 'have_family', 'have_pet', 'have_sibling'],
             'marital status': ['marital_status'],
             'living place': ['live_in_general','live_in_citystatecountry','place_origin','nationality'],
             'vehicle': ['have_vehicle'],
             'personal attribute': ['physical_attribute', 'misc_attribute','has_age','gender']}

class Utt2TopicWord(Dataset):
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
            topic_nli = json.load(open(os.path.join(self.args.data_dir, f'topic_nli/Utt2TopicWord_{self.split}.json')))
        else:
            topic_nli = json.load(open(os.path.join(topic_nli_dir, f'Utt2TopicWord_{self.split}.json')))

        return topic_nli

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



class Utt2TopicWordExceptJob(Utt2TopicWord):
    def _load_data(self):
        if self.args.data_dir:
            data_all = json.load(open(os.path.join(self.args.data_dir, f'topic_nli/Utt2TopicWord_{self.split}.json')))
        else:
            data_all = json.load(open(os.path.join(topic_nli_dir, f'Utt2TopicWord_{self.split}.json')))

        topic_nli = []
        if self.split == "train":
            for ele in data_all:
                if ele["gold_topic"] != 'job' and ele["gold_topic"] != "vehicle":
                    topic_nli.append(ele)
        else:
            topic_nli = data_all

        return topic_nli

class Utt2TopicWord_and_MNLI(Utt2TopicWord):
    def _load_data(self):
        if self.args.data_dir:
            topic_nli = json.load(open(os.path.join(self.args.data_dir, f'topic_nli/Utt2TopicWord_{self.split}.json')))
        else:
            topic_nli = json.load(open(os.path.join(topic_nli_dir, f'Utt2TopicWord_{self.split}.json')))

        if self.split == "train":
            convDict = {"entailment": "positive",
                        "contradiction": "negative",
                        "neutral": "neutral"}
            mnliData = []
            if self.args.data_dir:
                with open(os.path.join(self.args.data_dir, f'multinli_1.0/multinli_1.0_{self.split}.jsonl'), 'r') as json_file:
                    json_list = list(json_file)
            for json_str in json_list:
                ele = json.loads(json_str)
                mnliData.append({"sentence1": ele["sentence1"],
                                 "sentence2": ele["sentence2"],
                                 "label": convDict[ele["gold_label"]],
                                 "id": ele["pairID"]
                                 })
            topic_nli.extend(mnliData)

        return topic_nli

class Utt2TopicDesc(Utt2TopicWord):
    def __init__(self, args, split):
        super().__init__()
        self.label2id = {"negative": 0,
                         "neutral": 1,
                         "positive": 2}

    def _load_data(self):
        if self.args.data_dir:
            topic_nli = json.load(open(os.path.join(self.args.data_dir, f'topic_nli/Utt2TopicDesc_{self.split}.json')))
        else:
            topic_nli = json.load(open(os.path.join(topic_nli_dir, f'Utt2TopicDesc_{self.split}.json')))
        return topic_nli

class Utt2TopicDescExceptJob(Utt2TopicDesc):
    def _load_data(self):
        if self.args.data_dir:
            data_all = json.load(open(os.path.join(self.args.data_dir, f'topic_nli/Utt2TopicDesc_{self.split}.json')))
        else:
            data_all = json.load(open(os.path.join(topic_nli_dir, f'Utt2TopicDesc_{self.split}.json')))

        topic_nli = []
        if self.split == "train":
            for ele in data_all:
                if ele["gold_topic"] != 'job' and ele["gold_topic"] != "vehicle":
                    topic_nli.append(ele)
        else:
            topic_nli = data_all

        return topic_nli

import re
def parse_dial(sample):
    split_result = re.sub(
        r"[<](speaker1|speaker2)[>]", "</s>", sample.replace("<s>", "")
    ).split("</s>")
    split_result = [s for s in split_result if s.strip()]
    dialog = []
    for idx, r in enumerate(split_result):
        r = re.sub(r"\s+", " ", r.replace("\n", " ").replace("</", "").strip())
        dialog.append(r)
    return dialog

unseen_topic = ['health', 'life tips', 'today i learn', 'religion', 'politics']

class Utt2TopicWord_NewTopic(Utt2TopicWord):

    def _load_data(self):
        assert self.args.generated_data_type
        assert self.args.generator
        assert self.args.train_datafile
        generated_data = json.load(open(os.path.join(self.args.sess_dir, 'neuralpc',
                                                     self.args.train_datafile, self.args.generator,
                                                     self.args.generated_data_type + "-output.json")))
        examples = []
        for idx, ele in enumerate(generated_data):
            if "TurnLevel" in self.args.generated_data_type:
                ele["generated_dialog"] = parse_dial(ele["generated_dialog"].split("<context>")[1])
            rm_pad = []
            for utt in ele["generated_dialog"]:
                if not utt.startswith('<pad><pad>'):
                    rm_pad.append(utt)
            ele["cleansed_dialog"] = rm_pad

            for id_u, utt in enumerate(ele["cleansed_dialog"]):
                example = {"did": f"{self.args.generated_data_type}:dial_{idx}:utt_{id_u}",
                                "sentence1": utt}
                for q_topic in unseen_topic+list(topic_dict.keys()):
                    sent2 = f"the text is about the {q_topic}"
                    example["sentence2"] = sent2
                    example["query_topic"] = q_topic
                    example["label"] = "negative"

                    examples.append(example)

        return examples

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