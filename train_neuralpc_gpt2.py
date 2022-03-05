
import os, time, argparse, random, json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (AdamW,
                          get_linear_schedule_with_warmup,
                          AutoTokenizer)
from tensorboardX import SummaryWriter

from dataset import NeuralPCDataset4LM
from model import GPT2NeuralPC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # load train and dev sets
        self.train_set = NeuralPCDataset4LM(self.args, split="train")
        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.dev_set = NeuralPCDataset4LM(self.args, split="valid")
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.batch_size)

        # load model
        self.model = GPT2NeuralPC(self.args)
        self.tokenizer = self.model.tokenizer
        self.model_best_params = {}

    # save model checkpoint to local directory
    def save(self, global_step):
        ckpt_path = os.path.join(self.args.output_path, f"checkpoint-{global_step}")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        config = self.model.model.config.to_dict()
        json.dump(config, open(os.path.join(ckpt_path, 'config.json'), 'w'), indent=4)
        self.tokenizer.save_vocabulary(ckpt_path)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), os.path.join(ckpt_path, 'pytorch_model.bin'))
        print(f"*** Model checkpoint saved to {ckpt_path} ***")

    # calculate dev loss and update best model parameters
    def update_best(self, train_loss, global_step, step):
        dev_loss = self.dev()
        print(f"*** At Step {global_step}/{len(self.train_loader)}: ", \
              f"train loss {np.mean(train_loss) / step:.4f}, dev loss {dev_loss:.4f} ***")
        self.model_best_params = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if dev_loss < self.best_dev:
            self.best_dev = dev_loss
        self.save(global_step)

        return dev_loss

    def train(self):
        self.best_dev = 999999, 0

        tb_writer = SummaryWriter(self.args.output_path)
        params = []
        params.append({'params': self.model.model.parameters(), 'lr': self.args.lr})
        # optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        t_total = len(self.train_loader) // self.args.gradient_accumulation_steps * self.args.num_epochs
        optimizer = AdamW(params, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        self.best_dev = self.dev()
        print(f"*** Initial dev loss: {self.best_dev:.4f} ***")

        global_step = 0
        self.model.zero_grad()

        for epoch in range(self.args.num_epochs):
            train_loss = 0
            for step, batch in enumerate(tqdm(self.train_loader), start=1):
                self.model.train()
                loss = self.model.forward(batch)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                train_loss += loss.item()
                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    global_step += 1

                # evaluate dev loss every 500 steps
                if (step+1) % self.args.logging_steps == 0:
                    tb_writer.add_scalar("train/learning_rate", scheduler._last_lr[0], global_step)
                    tb_writer.add_scalar("train/loss", train_loss/step, global_step)

                if (step+1) % (self.args.dev_at_step) == 0:
                    dev_loss = self.update_best(train_loss, global_step, step)
                    tb_writer.add_scalar("eval/loss", dev_loss, global_step)

            _ = self.update_best(train_loss, global_step, step)

    # calculate dev loss and print intermediate outputs
    def dev(self):
        self.model.eval()
        tot_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dev_loader), start=1):
                loss = self.model.forward(batch)
                tot_loss += loss.item()
                if step % 100 == 0:
                    output = self.model.generate(batch[0][0])
                    print(f"dev sentence from: {batch[0][0]}")
                    print(f"dev gold resp: {batch[1][0]}")
                    print(f"dev sentence to: {output}")
        return tot_loss / step


def main(args):
    start_time = time.time()
    trainer = Trainer(args)
    print(f"**************start training**************")
    trainer.train()
    print(f"**************end training**************")
    print(f"training time: {(time.time()-start_time)/60:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile_name", type=str, required=True, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True, default=None)
    parser.add_argument("--model_name_or_path", type=str, default='gpt2')

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--dev_at_step", type=int, default=500)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    parser.add_argument("--max_target_len", type=int, default=1024)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0)

    parser.add_argument("--fp16", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=42)

    # parser.add_argument("--do_train", action='store_true', default=False)
    # parser.add_argument("--do_predict", action='store_true', default=False)
    # parser.add_argument("--n_gpu", type=int, default=1)
    # parser.add_argument("--per_gpu_eval_batch_size", type=int, default=4)
    # parser.add_argument("--max_seq_length", type=int, default=768)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    else:
        print(
            f"There is already {args.output_path}. Please double check the output path if you don't want to overwrite!")
    main(args)