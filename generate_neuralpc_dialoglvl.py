

import os
import re
import json
import math
import torch
from transformers import (AutoTokenizer,
                          BartTokenizer,
                          BartForConditionalGeneration, )
import argparse
from util.seed import set_seed
from dataset import NeuralPCDataset, NeuralPC_SPECIAL_TOKENS, NeuralPCTestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    set_seed(args.seed)

    model = BartForConditionalGeneration.from_pretrained(args.output_dir)
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained(args.output_dir)
    tokenizer.add_special_tokens(NeuralPC_SPECIAL_TOKENS)

    if not args.test_type:
        test_dataset = NeuralPCDataset(args, 'test', tokenizer)
    else:
        test_dataset = NeuralPCTestDataset(args, 'test', tokenizer)
    inputs = test_dataset.examples

    res_dialogs = []
    n_step = math.ceil(len(test_dataset) / args.batch_size)
    print(f"Start Collection using {args.output_dir}!")
    temp = 1.0 if not args.temperature else args.temperature
    print(
        "Num Beams: %d, Top k: %d, Top p: %.2f, Temperature: %.2f"
        % (args.num_beams, args.top_k, args.top_p, temp)
    )
    if args.greedy_ratio > 0.0:
        n_greedy_step = math.ceil(n_step * args.greedy_ratio)
    else:
        n_greedy_step = -1
    print("Greedy ratio: %.3f, Greedy step: %d" % (args.greedy_ratio, n_greedy_step))

    for i in range(n_step):
        curr_batch = inputs[i * args.batch_size: (i + 1) * args.batch_size]
        if i < n_greedy_step:
            top_k, top_p, temperature, num_beams = 0, 0.0, None, 1
        else:
            top_k, top_p, temperature, num_beams = (
                args.top_k,
                args.top_p,
                args.temperature,
                args.num_beams,
            )

        batch = test_dataset.collate_fn(curr_batch)
        input_ids = batch[0].to(device)
        outputs = model.generate(
            input_ids,
            decoder_start_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=num_beams,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            max_length=args.max_target_length,
            temperature=temperature,
            early_stopping=True,
        )

        dialogs = []
        for output in outputs.cpu().tolist():
            result = tokenizer.decode(output)
            split_result = re.sub(
                r"[<](speaker1|speaker2)[>]", "</s>", result.replace("<s>", "")
            ).split("</s>")
            split_result = [s for s in split_result if s.strip()]
            dialog = []
            for idx, r in enumerate(split_result):
                r = re.sub(r"\s+", " ", r.replace("\n", " ").replace("</", "").strip())
                dialog.append(r)
            dialogs.append(dialog)

        for b, d in zip(curr_batch, dialogs):
            res_dialogs.append({"id": b["did"],
                                "topic_flow": b["topic_flow"],
                                "generated_dialog": d})
        print("[%d/%d]" % (i, n_step))


    json.dump(
        res_dialogs,
        open(os.path.join(args.output_dir, args.output_file_name+".json"), "w"),
        indent=2
    )
    print(
        "\nAll done! The synthesized dialogues are saved at %s" % os.path.join(args.output_dir, args.output_file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile_name", type=str, required=True, default=None)
    parser.add_argument("--test_type", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--output_file_name", type=str, default="neuralpc-output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_domain", type=str, default=None)
    parser.add_argument("--greedy_ratio", type=float, default=0.0)
    parser.add_argument(
        "--include_missing_dontcare", action="store_true", default=False
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print("wrong checkpoint path!")
        assert 1==0

    if os.path.exists(os.path.join(args.output_dir, args.output_file_name)):
        print(
            f"There is already {args.output_file_name} in the {args.output_dir}. Please doule check the output path if you don't want to overwrite!")

    main(args)
