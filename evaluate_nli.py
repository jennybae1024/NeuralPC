# Evalutate NLI Classifier

import os, json, time
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from util.args import get_args
from dataset2 import dnliDataset, dnliExceptJob, DnliMnliDataset, DnliMnliExceptJob
from dataset3 import (Utt2TopicWord, Utt2TopicWordExceptJob, Utt2TopicWord_and_MNLI,
                      Utt2TopicDesc, Utt2TopicDescExceptJob, Utt2TopicWord_NewTopic)

from transformers import (
                AutoModelForSequenceClassification
)

dataset_class = {"dnliOriginal": dnliDataset,
                 "dnliExceptJob": dnliExceptJob,
                 "dnli_mnli_All": DnliMnliDataset,
                 "dnli_mnli_ExceptJob": DnliMnliExceptJob,
                 "utt2topicWord": Utt2TopicWord,
                 "utt2topicWord_ExceptJob": Utt2TopicWordExceptJob,
                 "utt2topicWord_mnli": Utt2TopicWord_and_MNLI,
                 "utt2topicDesc": Utt2TopicDesc,
                 "utt2topicDesc_ExceptJob": Utt2TopicDescExceptJob,
                 "utt2topicWord_newTopic": Utt2TopicWord_NewTopic
                 }



class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        # load test set
        self.model_path = self.args.model_name_or_path
        self.output_path = self.args.output_dir
        self.test_set = dataset_class[self.args.dataset_name](self.args, split="test")
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.batch_size, collate_fn=self.test_set.collate_fn)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.checkpoint).to(self.args.device)

    def evaluate(self, tag=None):
        ckpt = "0000"

        self.model.eval()
        gold_labels, preds = [], []
        probs = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_loader)):
                input_ids, attention_mask, labels = batch
                model_outputs = self.model(input_ids=input_ids.to(self.device),
                                           attention_mask=attention_mask.to(self.device),
                                           labels=labels.to(self.device))
                loss = model_outputs[0]
                gold_labels.extend(list(labels.detach().cpu().numpy()))
                preds.extend(list(np.argmax(model_outputs[1].detach().cpu().numpy(), 1)))
                probs.extend(torch.softmax(model_outputs[1], 1)[:, 2].detach().cpu().numpy())

        assert len(gold_labels) == len(preds)

        if self.args.generated_data_type:
            # when evaluate generated dialog from neuralPC generator
            with open(os.path.join(self.args.output_dir, f"{self.args.generated_data_type}-{self.args.generator}-{self.args.train_datafile}-ckpt{ckpt}_results.txt"), 'w') as fout:
                for l, pr in zip(gold_labels, preds):
                    fout.write("\t".join([str(l), str(pr)])+"\n")
            fout.close()
            with open(os.path.join(self.args.output_dir, f"{self.args.generated_data_type}-{self.args.generator}-{self.args.train_datafile}-ckpt{ckpt}_probs.npy"), 'wb') as fout:
                np.save(fout, probs)

        else:
            with open(os.path.join(self.args.output_dir, f"ckpt{ckpt}_results.txt"), 'w') as fout:
                for l, pr in zip(gold_labels, preds):
                    fout.write("\t".join([str(l), str(pr)])+"\n")
            fout.close()
            with open(os.path.join(self.args.output_dir, f"ckpt{ckpt}_probs.npy"), 'wb') as fout:
                np.save(fout, probs)

def main():
    args = get_args()

    args.mode = "eval"
    start_time = time.time()
    evaluator = Evaluator(args)
    print(f"**************Evaluation start**************")
    evaluator.evaluate()
    print(f"Evaluation time: {(time.time() - start_time) / 60:.2f} min")


if __name__ == '__main__':
    main()
