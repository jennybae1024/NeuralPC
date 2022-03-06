import os, time
import torch
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from dataset import NeuralPCDataset4LM, GPT2NeuralPC_SPECIAL_TOKENS
from model import GPT2NeuralPC

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device

        self.model = GPT2NeuralPC(args)
        ckpt = torch.load(os.path.join(args.checkpoint, "pytorch_model.bin"), map_location=args.device)
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        self.tokenizer.add_special_tokens(GPT2NeuralPC_SPECIAL_TOKENS)

        print(f"Loaded checkpoint from {self.args.checkpoint}")

        self.output_path = self.args.output_path

    def generate(self):
        test_set = NeuralPCDataset4LM(self.args, split="test")
        test_loader = DataLoader(test_set, batch_size=1)

        top_k, top_p, temperature, num_beams = (
            args.top_k,
            args.top_p,
            args.temperature,
            args.beam_size,
        )
        if args.top_k != 0:
            assert top_p == 0
            assert num_beams == 1
            decoding_key=f'topK_{top_k}'
        elif top_p != 0:
            assert num_beams == 1
            decoding_key=f"topP_{top_p}"
        else:
            assert num_beams>1
            decoding_key=f"beamSearch_{num_beams}"

        ckpt = self.args.checkpoint.split("/")[-1].split("-")[1]
        outfile_name = f"{self.output_path}/ckpt{ckpt}-{decoding_key}-generations.txt"

        self.model.eval()
        with open(outfile_name, 'w') as fout:
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(test_loader)):
                    goal = batch[0][0]
                    result = self.model.generate(batch[0][0], top_k, top_p, num_beams)
                    fout.write("\t".join([goal, result])+"\n")
        fout.close()

    def evaluate(self):
        test_set = NeuralPCDataset4LM(self.args, split="valid")
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size)

        self.model.eval()
        tot_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(test_loader)):
                loss = self.model.forward(batch)
                tot_loss += loss.item()
                if step % 100 == 0:
                    output = self.model.generate(batch[0][0])
                    print(f"dev sentence from: {batch[0][0]}")
                    print(f"dev gold resp: {batch[1][0]}")
                    print(f"dev sentence to: {output}")
        print(f"Evaluation PPL: {tot_loss/step}")
        return tot_loss / step

def main(args):
    evaluator = Evaluator(args)
    start_time = time.time()
    if args.do_evaluate:
        print(f"**************Evaluation start**************")
        mode = "Evaluation"
        evaluator.evaluate()
    elif args.do_predict:
        print(f"**************Generation start**************")
        evaluator.generate()
        mode = "Generation"
    print(f"{mode} time: {(time.time() - start_time) / 60:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_evaluate", action='store_true', default=False)
    parser.add_argument("--do_predict", action='store_true', default=False)
    parser.add_argument("--datafile_name", type=str, required=True, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, default=None)
    parser.add_argument("--output_path", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_target_len", type=int, default=512)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_missing_dontcare", action="store_true", default=False
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.checkpoint):
        print("wrong checkpoint path!")
        assert 1==0

    main(args)
