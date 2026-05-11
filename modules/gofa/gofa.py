# example code for running inference with fine-tuned checkpoint
from typing import Optional

import numpy as np
import torch
from dataclasses import dataclass, field
from transformers import MistralConfig

from modules.gofa.gofa_icae import MistralICAE
from collections import OrderedDict
from safetensors.torch import load_file
from modules.utils import safe_download_hf_file

###################################################################
#                 Configurations                                  #
###################################################################


class GOFAMistralConfig(MistralConfig):
    def __init__(self, dim=4096, num_layers=6, mem_token=128, head=32, add_self_loops=True, dropout=0.0,
                 llama_dtype=torch.float16, gnn_hidden_act="relu", gnn_mlp_type="gp", gnn_type="index",
                 position_encoding="none", pretraining_tp=0, gating=True, interleave=True, mp_att="concat",
                 trainable_layer=5, fuse_type="interleave", model_parallel=False, model_parallel_devices=None,
                 model_parallel_splits=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mem_token = mem_token
        self.head = head
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.num_layers = num_layers
        self.llama_dtype = llama_dtype
        self.gnn_hidden_act = gnn_hidden_act
        self.gnn_mlp_type = gnn_mlp_type
        self.gnn_type = gnn_type
        self.pretraining_tp = pretraining_tp
        self.position_encoding = position_encoding
        self.interleave = interleave
        self.gating = gating
        self.mp_att = mp_att
        self.trainable_layer = trainable_layer
        self.fuse_type = fuse_type
        self.model_parallel = model_parallel
        self.model_parallel_devices = model_parallel_devices or ["cuda:0", "cuda:1"]
        self.model_parallel_splits = model_parallel_splits or [16]


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    lora_r: int = field(default=512, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    mem_size: int = field(default=128, metadata={"help": "Memory size"}, )
    dec_lora: bool = field(default=False, metadata={"help": "Whether using lora in the decoder LLM"})
    checkpoint_dir: str = field(default="./cache_data/model/")


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=False)
    model_max_length: int = field(default=512,
        metadata={"help": "Maximum sequence length per node. Sequences will be right padded (and possibly truncated)."}, )
    fixed_mem_size: int = field(default=128, metadata={"help": "Enalbing the fixed mem size."}, )
    mean_compression_rate: int = field(default=4, metadata={"help": "Mean compression rate; default=4"}, )
    min_tokens_for_lm: int = field(default=64, metadata={"help": "Minimum tokens for lm objective learning"}, )
    leave_tokens_for_lm: int = field(default=8, metadata={"help": "Leave some tokens without loss for lm objective"}, )
    lm_ratio: float = field(default=0.0, metadata={"help": "Ratio for LM training."}, )
    add_special_token_for_lm: bool = field(default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"}, )
    restore_from: str = field(default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"})


###################################################################
#                 Model                                           #
###################################################################


class GOFAMistral(torch.nn.Module):
    def __init__(self, transformer_args):
        super().__init__()
        model_args, training_args, gofa_args = transformer_args
        model = MistralICAE(model_args, training_args, gofa_args)  # restored llama2-7b-chat model
        dir = safe_download_hf_file("sggetao/icae", "mistral_7b_ft_icae.safetensors", model_args.checkpoint_dir, repo_type=None)
        state_dict = load_file(dir)  # change the path for your model
        new_state_dict = OrderedDict()
        for layer_name, weight in state_dict.items():
            new_state_dict[layer_name.replace("default", "encadapt")] = weight
        model.load_state_dict(new_state_dict, strict=False)
        # model.merge_lora()
        self.model_args = model_args
        self.dec_lora = model_args.dec_lora
        self.mem_tokens = list(range(model.vocab_size, model.vocab_size + model_args.mem_size))
        self.mem_size = model_args.mem_size
        self.model = model
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.left_tokenizer.pad_token = self.model.left_tokenizer.bos_token
        for param in self.model.icae.parameters():
            param.requires_grad = False
        for param in self.model.icae.get_base_model().model.g_layers.parameters():
            param.requires_grad = True
        if self.dec_lora:
            for name, param in self.model.icae.named_parameters():
                if "default" in name:
                    param.requires_grad = True

    def get_tokenizer(self):
        return self.model.tokenizer

    def train_mode(self):
        self.model.icae.set_adapter("encadapt")
        for param in self.model.icae.parameters():
            param.requires_grad = False

    def load_pretrained(self, pretrained_path=None):
        if pretrained_path is None:
            pretrained_path = safe_download_hf_file("WFRaain/GOFA", "mistral_qamag03_best_ckpt.pth", self.model_args.checkpoint_dir,
                                        repo_type=None)
        self.load_partial(pretrained_path)

    def save_partial(self, save_dir):
        """
        Save the GNN and lora weight (if available).
        """
        state_dict = self.model.icae.get_base_model().model.g_layers.state_dict()
        full_state_dict = self.state_dict()
        for k in full_state_dict:
            if "default" in k:
                state_dict[k] = full_state_dict[k]
        torch.save(state_dict, save_dir)

    def load_partial(self, load_dir):
        """
        Load the GNN and lora weight (if available).
        """
        state_dict = torch.load(load_dir, map_location="cpu")
        missing_keys, _ = self.model.icae.get_base_model().model.g_layers.load_state_dict(state_dict, strict=False)
        print("GNN module is missing the following keys:", missing_keys)
        missing_keys, _ = self.load_state_dict(state_dict, strict=False)

    def forward(self, g):
        """
        Encode the graph and generate logits for answer tokens.
        """
        g.num_node_feat = g.x.shape[0]
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            text_inputs = np.concatenate([g.x, g.edge_attr], axis=0)
        else:
            text_inputs = g.x
        text_inputs = text_inputs.tolist()
        llm_output = self.encode(text_inputs, graph=g, partial_grad=True)
        emb = llm_output[:g.node_map.size(-1)]
        if not hasattr(g, "answer"):
            raise ValueError("Forward stage graph should contain answer.")
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        # Legacy hard coding TODO: remove when TAGLAS is fixed.
        prompt_input_texts = ["" if (p.startswith("Please complete the sentence of the node") or p == "") else p for p
                              in prompt_texts]
        emb = emb[g.question_index]
        answer_logits, answer_id, masks = self.decode(answer_texts, emb, prompt=prompt_input_texts)
        return answer_logits, answer_id, masks, answer_texts

    def generate(self, g, max_length=128):
        """
        Autoregressively generate tokens.
        """
        g.num_node_feat = g.x.shape[0]
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            text_inputs = np.concatenate([g.x, g.edge_attr], axis=0)
        else:
            text_inputs = g.x
        text_inputs = text_inputs.tolist()
        llm_output = self.encode(text_inputs, graph=g, partial_grad=True)
        emb = llm_output[:g.node_map.size(-1)]
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        prompt_input_texts = ["" if (p.startswith("Please complete the sentence of the node") or p == "") else p for p
                              in prompt_texts]
        emb = emb[g.question_index]
        generated_text = self.infer(emb, prompt=prompt_input_texts, max_length=max_length)
        return generated_text

    def encode(self, data, graph=None, partial_grad=None):
        cur_device = self.model.icae.get_base_model().model.embed_tokens.weight.device
        batch_size = len(data)
        text_output = \
        self.model.tokenizer(data, truncation=True, max_length=self.model.training_args.model_max_length, padding=False,
                             return_attention_mask=False)["input_ids"]

        text_output = [t + self.mem_tokens for t in text_output]
        text_output = {"input_ids": text_output}
        text_output = self.model.tokenizer.pad(text_output, padding=True, return_tensors="pt")["input_ids"].to(
            cur_device)
        mem_mask = text_output >= self.model.vocab_size

        mem_mask = mem_mask.to(cur_device)
        autoencoder_input_embedding = self.model.tokens_to_embeddings(text_output)

        # Use ICAE lora only in the encoder.
        self.model.icae.set_adapter("encadapt")
        self.model.icae.enable_adapter_layers()
        for name, param in self.model.icae.named_parameters():
            if "encadapt" in name:
                param.requires_grad = False
        compress_outputs = self.model.icae(inputs_embeds=autoencoder_input_embedding, output_hidden_states=True,
                                           graph=graph, mem_mask=mem_mask, partial_grad=partial_grad, map_node=True)
        self.model.icae.disable_adapter_layers()
        compress_outputs = compress_outputs.hidden_states[-1]

        if graph is not None:
            node_emb = compress_outputs[:len(graph.node_map)]
            map_mem_mask = mem_mask[:graph.num_node_feat][graph.node_map]
            memory_embedding = node_emb[map_mem_mask].view(len(node_emb), self.mem_size, -1)
        else:
            memory_embedding = compress_outputs[mem_mask].view(batch_size, self.mem_size, -1)
        return memory_embedding

    def decode(self, data, mem_embs, graph=None, prompt=None):
        cur_device = self.model.icae.get_base_model().model.embed_tokens.weight.device
        prompt_output = self.model.tokenizer(data, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        prompt_output = [p + [self.model.tokenizer.eos_token_id] if len(p) < self.model.training_args.model_max_length else p[:self.model.training_args.model_max_length] for p in prompt_output]
        original_prompt_output = prompt_output

        if prompt is None:
            prompt = [""] * len(data)
        prompt_input = self.model.left_tokenizer(prompt, add_special_tokens=False, padding=False, truncation=True, max_length=512)["input_ids"]
        batch_size = len(prompt_input)

        # For Mistral, decode contains: prefix, memory slots and suffix
        prompt_left_ids = [[1, 733, 16289, 28793] if len(a) > 0 else [] for a in prompt_input]
        prompt_right_ids = [[self.model.ft_token_id] + a + [733, 28748, 16289, 28793] if len(a) > 0 else a for a in
                            prompt_input]
        prompt_ids = [a + [self.model.tokenizer.pad_token_id] * self.mem_size + b + c for a, b, c in
                      zip(prompt_left_ids, prompt_right_ids, prompt_output)]
        prompt_mask = [
            [False] * (len(prompt_left_ids[i]) + self.mem_size - 1 + len(prompt_right_ids[i])) + [True] * len(
                prompt_output[i]) + [False] for i in range(batch_size)]

        answer_prompt = torch.cat([torch.tensor(p, dtype=torch.long) for p in prompt_output], dim=-1).to(
            cur_device)

        prompt_output = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
        prompt_output = self.model.tokenizer.pad(prompt_output, padding=True, return_tensors="pt")
        prompt_answer_ids = prompt_output["input_ids"].to(cur_device)
        prompt_answer_embs = self.model.tokens_to_embeddings(prompt_answer_ids)

        mem_mask = [[False] * len(prompt_left_ids[i]) + [True] * self.mem_size + [False] * (
                len(prompt_output["input_ids"][i]) - len(prompt_left_ids[i]) - self.mem_size) for i in
                    range(batch_size)]
        prompt_mask = [
            [False] * (len(prompt_left_ids[i]) + self.mem_size - 1 + len(prompt_right_ids[i])) + [True] * len(
                original_prompt_output[i]) + [False] * (1 + len(prompt_output["input_ids"][i]) - len(prompt_ids[i])) for
            i in range(batch_size)]

        prompt_answer_embs[torch.tensor(mem_mask, device=cur_device)] = mem_embs.to(cur_device).view(-1, mem_embs.size()[-1])

        target_mask = torch.tensor(prompt_mask, dtype=torch.long, device=cur_device).to(torch.bool)

        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        output_emb = self.model.icae(inputs_embeds=prompt_answer_embs).logits

        return output_emb, answer_prompt, target_mask

    def infer(self, mem_embs, graph=None, prompt=None, max_length=128):
        cur_device = self.model.icae.get_base_model().model.embed_tokens.weight.device

        if prompt is None:
            prompt = [""] * len(mem_embs)
        prompt_input = self.model.tokenizer(prompt, add_special_tokens=False, padding=False)["input_ids"]
        batch_size = len(prompt_input)

        prompt_left_ids = [[1, 733, 16289, 28793] if len(a) > 0 else [] for a in prompt_input]

        prompt_right_ids = [[self.model.ft_token_id] + a + [733, 28748, 16289, 28793] if len(a) > 0 else a for a in
                            prompt_input]

        mem_mask = [[False] * len(prompt_left_ids[i]) + [True] * self.mem_size + [False] * len(prompt_right_ids[i]) for
                    i in range(batch_size)]
        att_mask = [[True] * (len(prompt_left_ids[i]) + self.mem_size + len(prompt_right_ids[i])) for i in
                    range(batch_size)]
        prompt_ids = [prompt_left_ids[i] + [self.model.tokenizer.pad_token_id] * self.mem_size + prompt_right_ids[i] for
                      i in range(batch_size)]

        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": mem_mask},
                                                         padding=True, return_tensors="pt")
        mem_mask = input_prompt_ids["attention_mask"].to(device=cur_device, dtype=torch.bool)

        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": att_mask},
                                                         padding=True, return_tensors="pt")
        prompt_ids = input_prompt_ids["input_ids"]
        att_mask = input_prompt_ids["attention_mask"].to(device=cur_device)

        prompt_answer_ids = prompt_ids.to(device=cur_device, dtype=torch.long)
        prompt_answer_embs = self.model.tokens_to_embeddings(prompt_answer_ids)
        prompt_answer_embs[mem_mask] = mem_embs.to(cur_device).view(-1, mem_embs.size()[-1])

        decode_embed = prompt_answer_embs
        output = decode_embed.clone()

        generate_text = []
        eos_reached = torch.zeros(len(output), dtype=torch.bool).to(output.device)

        past_key_values = None
        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        for i in range(max_length):
            out = self.model.icae(inputs_embeds=output, attention_mask=att_mask, past_key_values=past_key_values,
                                 use_cache=True)

            logits = out.logits[:, -1, :self.model.vocab_size - 1]

            past_key_values = out.past_key_values

            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            eos_reached = torch.logical_or(eos_reached, (next_token_id == self.model.tokenizer.eos_token_id).view(-1))

            # eos_reached = torch.logical_or(eos_reached, (next_token_id==self.model.tokenizer.bos_token_id).view(-1))

            # eos_reached = torch.logical_or(eos_reached, (next_token_id>=32000).view(-1))

            next_token_for_embed = next_token_id.to(cur_device)
            output = self.model.icae.get_base_model().model.embed_tokens(next_token_for_embed)

            generate_text.append(next_token_id.view(-1, 1))
            att_mask = torch.cat(
                [att_mask, torch.ones((len(att_mask), 1), dtype=att_mask.dtype, device=att_mask.device)], dim=-1)

            if torch.all(eos_reached):
                break

        generate_text = torch.cat(generate_text, dim=-1)
        generate_text[generate_text >= 32000] = 1

        generated_text = self.model.tokenizer.batch_decode(generate_text)

        return generated_text
