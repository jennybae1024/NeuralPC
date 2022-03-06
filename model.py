import re
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from dataset import GPT2NeuralPC_SPECIAL_TOKENS


class GPT2NeuralPC(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = self.args.device

        # load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.add_special_tokens(GPT2NeuralPC_SPECIAL_TOKENS)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch):
        goals = batch[0]
        dialogs = batch[1]
        bz = len(dialogs)
        input_ids = [torch.LongTensor([
            [self.bos_token_id]
            + self.tokenizer.encode(goals[i])
            + [self.sep_token_id]
            + self.tokenizer.encode(dialogs[i])
            + [self.eos_token_id]
        ]).squeeze(0) for i in range(bz)]

        input_ids = pad_sequence(input_ids, True, padding_value=self.tokenizer.pad_token_id).long().to(self.device)
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = [torch.LongTensor([
            (len(self.tokenizer.encode(goals[i])) + 2) * [-100]
            + self.tokenizer.encode(dialogs[i])
            + [self.eos_token_id]
        ]).squeeze(0) for i in range(bz)]

        labels = pad_sequence(labels, True, padding_value=-100).long().to(self.device)
        output = self.model(input_ids=input_ids.to(self.device),
                            attention_mask=attention_mask.to(self.device).half(),
                            labels=labels.to(self.device))
        loss = output[0]
        return loss

    def generate(self, batch, top_k=0, top_p=0, num_beams=1):
        context = batch[0]

        input_ids = torch.LongTensor([
            [self.bos_token_id]
            + self.tokenizer.encode(context)
            + [self.sep_token_id]
        ]).to(self.device)



        model_output = self.model.generate(input_ids=input_ids.to(self.device),
                                           eos_token_id=self.tokenizer.eos_token_id,
                                           pad_token_id=self.tokenizer.pad_token_id,
                                           num_beams=num_beams,
                                           do_sample=True,
                                           top_k=top_k,
                                           top_p=top_p,
                                           max_length=self.args.max_target_len,
                                           early_stopping=True)

        result = self.tokenizer.decode(model_output.cpu()[0])
        result = result.split("<sep>", 1)[1]

        return result

