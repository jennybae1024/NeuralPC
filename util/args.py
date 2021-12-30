import json
import torch
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    # basic settings
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name_or_path", type=str, default='roberta-base')
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--datafile_name", type=str)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output_dir", type=str)

    # training hyperparams
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dev_at_step", type=int, default=500)
    parser.add_argument("--fp16", type=bool, default=False)

    # evaluate generated dialog from NeuralPC
    parser.add_argument("--gen_data", type=str, default=None, help="use if data is generated dialog from neuralpc")
    parser.add_argument("--train_datafile", type=str, default=None, help="train dataset type of generator")
    parser.add_argument("--generator", type=str, default=None, help="generator")
    parser.add_argument("--generated_data_type", type=str, default=None, help="neural test type 1 or 2 or 3")

    # sampling hyperparams
    parser.add_argument("--num_generate", type=int, default=1)
    parser.add_argument("--max_sent_len", type=int, default=64)
    parser.add_argument("--min_sent_len", type=int, default=3)
    parser.add_argument("--regenerate_try", type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--no_sample', type=bool, default=False)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print(f"Device: {args.device}, # GPU: {args.n_gpu}")

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    return args
