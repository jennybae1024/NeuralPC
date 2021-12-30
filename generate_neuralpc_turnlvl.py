

import os
import re
import json
import math
import torch
from transformers import (AutoTokenizer,
                          BartTokenizer,
                          BartForConditionalGeneration, )
import argparse
from tqdm import tqdm
from util.seed import set_seed
from dataset import NeuralPCDataset, NeuralPC_SPECIAL_TOKENS, NeuralPCTestDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(data):
    split_result = re.sub(
                r"[<](speaker1|speaker2)[>]", "</s>", data.replace("<s>", "")
            ).split("</s>")
    split_result = [s.strip() for s in split_result if s.strip()]
    return split_result

def main(args):
    set_seed(args.seed)

    model = BartForConditionalGeneration.from_pretrained(args.output_dir)
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained(args.output_dir)
    tokenizer.add_special_tokens(NeuralPC_SPECIAL_TOKENS)

    # test_dataset = NeuralPCDataset(args, 'test', tokenizer)
    if not args.test_type:
        test_dataset = NeuralPCDataset(args, 'test', tokenizer)
    else:
        test_dataset = NeuralPCTestDataset(args, 'test', tokenizer)

    res_dialogs = []
    n_step = math.ceil(len(test_dataset) / args.batch_size)
    print(f"Start Collection using {args.output_dir}!")
    temp = 1.0 if not args.temperature else args.temperature
    print(
        "Num Beams: %d, Top k: %d, Top p: %.2f, Temperature: %.2f"
        % (args.num_beams, args.top_k, args.top_p, temp)
    )


    for sample in tqdm(test_dataset.data):
        goal_text = sample["goal"]
        turn = 0
        with torch.no_grad():
            new_dialogs = []
            while turn < args.max_turns:
                input_text = goal_text + "".join(new_dialogs[-args.turn_window_size:])
                input_ids = torch.LongTensor([tokenizer(input_text)["input_ids"]])
                outputs = model.generate(
                    input_ids.to(device),
                    decoder_start_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    num_beams=1,
                    do_sample=True,
                    #         top_k=args.top_k,
                    top_p=args.top_p,
                    max_length=args.max_target_length,
                    temperature=args.temperature,
                    early_stopping=True,
                )
                for output in outputs.cpu().tolist():
                    curr_turn = tokenizer.decode(output)
                    curr_turn = clean_text(curr_turn)
                    if len(curr_turn) == 2:
                        new_dialogs.append(" <speaker1> " + curr_turn[0] + " <speaker2> " + curr_turn[1])
                    elif len(curr_turn) == 1:
                        new_dialogs.append(" <speaker1> " + curr_turn[0])
                        break
                    else:
                        break
                    turn += 1

        res_dialogs.append({"id": sample["id"],
                            "topic_flow": sample["topic_flow"],
                            "speaker1_persona": sample["speaker1_persona"],
                            "speaker2_persona": sample["speaker2_persona"],
                            "generated_dialog": input_text})

    json.dump(
        res_dialogs,
        open(os.path.join(args.output_dir,
                  args.output_file_name+f"-maxTurn{args.max_turns}_window{args.turn_window_size}-output.json"), "w"),
        indent=2
    )
    print(
        "\nAll done! The synthesized dialogues are saved at %s" % os.path.join(args.output_dir, args.output_file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile_name", type=str, required=True, default=None)
    parser.add_argument("--test_type", type=str, default=None)
    parser.add_argument("--max_turns", type=int, default=7)
    parser.add_argument("--turn_window_size", type=int, default=7)
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
