from collections import namedtuple

import torch

from modules.gofa.gofa import GOFAMistral
from modules.llm.mistral import MistralHelper
from modules.mplm.mplm import MPLM


def add_self_loop(data ,**kwargs):
    edge_index = data.edge_index
    edge_index = torch.cat([edge_index, torch.arange(data.num_nodes, device=edge_index.device, dtype=torch.long).repeat(2, 1)], dim=-1)
    edge_attr = data.edge_attr
    # edge_attr = np.concatenate([edge_attr, np.array(["This is an edge connecting a node to itself."])])
    edge_map = data.edge_map
    edge_map = torch.cat([edge_map, torch.tensor([len(edge_attr) - 1] * data.num_nodes, device=edge_map.device)])
    data.edge_index = edge_index
    data.edge_map = edge_map
    data.edge_attr = edge_attr
    return data

def print_fixed_length(text, line_width=120):
    t_pointer = 0
    while t_pointer < len(text):
        t_pointer += line_width
        t_cur_text = text[t_pointer - line_width:t_pointer]
        print(t_cur_text)


def print_text_side_by_side(text_1, text_2, line_width=120, space=10):
    t1_len = len(text_1)
    t2_len = len(text_2)
    seg_width = int((line_width-space)/2)
    t1_pointer = 0
    t2_pointer = 0
    text_1 = text_1.replace('\n', ' ').replace('\r', ' ')
    text_2 = text_2.replace('\n', ' ').replace('\r', ' ')
    while t1_pointer<t1_len or t2_pointer<t2_len:
        t1_pointer += seg_width
        t2_pointer += seg_width
        t1_cur_text = text_1[t1_pointer-seg_width:t1_pointer]
        t2_cur_text = text_2[t2_pointer-seg_width:t2_pointer]
        t1_cur_text = t1_cur_text + " "*(seg_width-len(t1_cur_text))
        t2_cur_text = t2_cur_text + " "*(seg_width-len(t2_cur_text))
        full_text = t1_cur_text + " "*space + t2_cur_text
        print(full_text)


def identity(x):
    return x


class GOFA(torch.nn.Module):
    def __init__(self, transformer_args, mode="autoencoder", base_llm="mistral7b", save_dir=""):
        super().__init__()

        self.mode = mode
        self.save_dir = save_dir

        if base_llm == 'mistral7b':
            self.llm_model = GOFAMistral(transformer_args)
        elif base_llm == 'mistral7blora':
            self.llm_model = MistralHelper(transformer_args)
        elif base_llm == 'mistral7bmplmsparse':
            self.llm_model = MPLM(transformer_args)
        else:
            raise NotImplementedError(base_llm + " is not supported. Please choose from: llama7b, mistral7b,")

        if mode == "decode":
            self.process = self.auto_encode_decode
        elif mode == "generate":
            self.process = self.generate
        else:
            # TODO: not implemented
            raise NotImplementedError(mode + " mode not implemented")

    def auto_encode_decode(self, g):
        answer_logits, answer_id, masks, answer_texts = self.llm_model(g)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=answer_logits[masks][:, :32000], pred_text=self.logit_to_text(answer_logits, masks),
                           answer_id=answer_id, answer=answer_texts)

    def generate(self, g):
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        generated_text = self.llm_model.generate(g)
        for i, txt in enumerate(generated_text):
            print_fixed_length("question: " + prompt_texts[i])
            print("-"*120)
            print_text_side_by_side("target: "+answer_texts[i], "gen: "+generated_text[i])
            print("="*120)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=torch.randn([1, 32132]), pred_text=generated_text, answer_id=torch.tensor([1]),
                           answer=answer_texts)

    def forward(self, g):
        return self.process(g)

    def save_partial(self, save_dir):
        self.llm_model.save_partial(save_dir)

    def load_partial(self, load_dir):
        self.llm_model.load_partial(load_dir)

    def logit_to_text(self, logits, masks):
        tokenizer = self.llm_model.get_tokenizer()
        if len(logits.size()) == 2:
            logits = logits.unsqueeze(0)
        decoded_texts = []
        for i in range(logits.size(0)):
            sample_logit = logits[i]
            sample_mask = masks[i]
            sample_logit = sample_logit[sample_mask]
            token_ids = sample_logit[:, :32000].argmax(dim=-1).unsqueeze(0)
            token_ids[token_ids >= 32000] = 1
            sample_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_texts.extend(sample_text)
        return decoded_texts

