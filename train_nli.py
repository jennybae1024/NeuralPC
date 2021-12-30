
# DNLI Scorer Trainer

import os, sys, json, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset2 import dnliDataset, dnliExceptJob, DnliMnliDataset, DnliMnliExceptJob
from dataset3 import Utt2TopicWord, Utt2TopicWordExceptJob, Utt2TopicWord_and_MNLI, Utt2TopicDesc, Utt2TopicDescExceptJob
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AdamW,
                          get_linear_schedule_with_warmup)
from util.args import get_args
from tqdm import tqdm
from tensorboardX import SummaryWriter

data_class = {"dnliOriginal": dnliDataset,
              "dnliExceptJob": dnliExceptJob,
              "dnli_mnli_All": DnliMnliDataset,
              "dnli_mnli_ExceptJob": DnliMnliExceptJob,
              "utt2topicWord": Utt2TopicWord,
              "utt2topicWord_ExceptJob": Utt2TopicWordExceptJob,
              "utt2topicWord_mnli": Utt2TopicWord_and_MNLI,
              "utt2topicDesc": Utt2TopicDesc,
              "utt2topicDesc_ExceptJob": Utt2TopicDescExceptJob,
              }


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device

        # load train and dev sets
        self.train_set = data_class[self.args.dataset_name](self.args, split="train")
        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, collate_fn=self.train_set.collate_fn, shuffle=True, drop_last=True)
        self.dev_set = data_class[self.args.dataset_name](self.args, split="dev")
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.batch_size,  collate_fn=self.dev_set.collate_fn)

        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_name_or_path,
                                                                        num_labels=self.train_set.num_labels).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.model_best_params = {}

    def save(self, step):
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        model_to_save.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        torch.save(self.args, os.path.join(self.args.output_dir, "training_args.bin"))

        print(f"*** Model checkpoint-{step} saved to {self.args.output_dir} ***")

        with open(os.path.join(self.args.output_dir, 'ckpt_log.txt'), 'a') as file:
            file.write(f"*** Model checkpoint-{step} saved to {self.args.output_dir} ***\n")

    def update_best(self, epoch, step, train_loss, global_step, epoch_end = None):
        dev_loss, dev_acc = self.dev()
        if epoch_end:
            print(f"*** Epoch {epoch} Step {step}/{len(self.train_loader)}: ", \
                  f"train loss {train_loss :.4f}, dev loss {dev_loss:.4f}, dev acc {dev_acc:.4f} ***")
        else:
            print(f"*** Epoch {epoch} Step {step}/{len(self.train_loader)}: ", \
              f"train loss {train_loss / step:.4f}, dev loss {dev_loss:.4f}, dev acc {dev_acc:.4f} ***")
        self.model_best_params = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if self.best_dev < dev_acc:
            self.best_dev = dev_acc
            self.save(global_step)

        return dev_loss, dev_acc

    def train(self):
        self.best_dev = 0

        tb_writer = SummaryWriter(self.args.output_dir)
        params = []
        params.append({'params': self.model.parameters(), 'lr': self.args.lr})
        # optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        t_total = len(self.train_loader) // self.args.gradient_accumulation_steps * self.args.num_epochs
        optimizer = AdamW(params, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        init_loss, self.best_dev = self.dev()
        print(f"*** Initial dev loss: {init_loss:.4f}, dev acc {self.best_dev:.4f} ***")

        global_step = 0
        self.model.zero_grad()

        for epoch in range(self.args.num_epochs):
            train_loss = 0
            for step, batch in enumerate(tqdm(self.train_loader), start=1):
                self.model.train()
                input_ids, attention_mask, labels = batch
                model_outputs = self.model(input_ids=input_ids.to(self.device),
                                           # token_type_ids=token_type_ids.to(self.device),
                                            attention_mask=attention_mask.to(self.device),
                                            labels=labels.to(self.device))
                loss = model_outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                train_loss += loss.item()
                global_step += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()

                # evaluate dev loss every 500 steps
                if step % (self.args.dev_at_step) == 0:
                    dev_loss, dev_acc = self.update_best(epoch, step, train_loss, global_step)
                    tb_writer.add_scalar("eval_loss", dev_loss, global_step)
                    tb_writer.add_scalar("eval_acc", dev_acc, global_step)
                    tb_writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("train/loss", train_loss/step, global_step)

            # scheduler.step()
            train_loss /= step
            tb_writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("train/loss", train_loss, global_step)

            dev_loss, dev_acc = self.update_best(epoch, step, train_loss, global_step, epoch_end=True)
            tb_writer.add_scalar("eval_loss", dev_loss, global_step)
            tb_writer.add_scalar("eval_acc", dev_acc, global_step)

    # calculate dev loss and print intermediate outputs
    def dev(self):
        self.model.eval()
        tot_loss = 0
        gold_labels = []
        preds = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dev_loader), start=1):
                input_ids, attention_mask, labels = batch
                # len()
                # print(input_ids)
                # print(token_type_ids)
                # print(attention_mask)
                # print(labels)
                # 1/0
                model_outputs = self.model(input_ids=input_ids.to(self.device),
                                           # token_type_ids=token_type_ids.to(self.device),
                                            attention_mask=attention_mask.to(self.device),
                                            labels=labels.to(self.device))
                loss = model_outputs[0]
                gold_labels.extend(labels.detach().cpu().numpy())
                preds.extend(np.argmax(model_outputs[1].detach().cpu().numpy(), 1))
                tot_loss += loss.item()

        eval_loss = tot_loss / step
        eval_acc = sum(np.array(gold_labels)==np.array(preds))/len(gold_labels)

        return eval_loss, eval_acc

def main():
    args = get_args()
    start_time = time.time()
    trainer = Trainer(args)
    print(f"**************start training**************")
    trainer.train()
    print(f"**************end training**************")
    print(f"training time: {(time.time()-start_time)/60:.2f} min")

if __name__ == '__main__':
    main()
