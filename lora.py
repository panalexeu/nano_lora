from typing import Literal
from dataclasses import dataclass, field

import torch 
import torch.nn as nn 
from torch.nn import functional as F

from model import GPT

@dataclass 
class LoRAConfig: 
    rank: int = 1
    model_type: Literal['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'] = 'gpt2'
    adapts: set[Literal['wq', 'wk', 'wv']] = field(default_factory=set(['wv']))
    block_size: int = 1024

class LoRAGPT(nn.Module): 
    def __init__(
        self, 
        cfg: LoRAConfig
    ):
        super().__init__()

        self.cfg = cfg 
        self.fmodel = GPT.from_pretrained(self.cfg.model_type)
        self.fmodel.requires_grad_(False)  
        self.embed = self.fmodel.config.n_embd

        assert 1 <= self.cfg.rank <= self.embed 
        assert 1 <= len(self.cfg.adapts) <= 3

        # low rank matrices init 
        self.wq_A = self._init_A() if 'wq' in self.cfg.adapts else None
        self.wq_B = self._init_B() if 'wq' in self.cfg.adapts else None
        self.wk_A = self._init_A() if 'wk' in self.cfg.adapts else None
        self.wk_B = self._init_B() if 'wk' in self.cfg.adapts else None
        self.wv_A = self._init_A() if 'wv' in self.cfg.adapts else None
        self.wv_B = self._init_B() if 'wv' in self.cfg.adapts else None

    def _init_A(self) -> nn.ModuleList:
        return nn.ModuleList([nn.Linear(in_features=self.embed, out_features=self.cfg.rank, bias=False) for _ in range(self.fmodel.config.n_layer)])

    def _init_B(self,) -> nn.ModuleList: 
        return nn.ModuleList([nn.Linear(in_features=self.cfg.rank, out_features=self.embed, bias=False) for _ in range(self.fmodel.config.n_layer)])

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.cfg.block_size, f"Cannot forward sequence of length {t}, block size is only {self.cfg.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.fmodel.transformer.wte(idx)
        pos_emb = self.fmodel.transformer.wpe(pos) 
        x = self.fmodel.transformer.drop(tok_emb + pos_emb)
        for block in self.fmodel.transformer.h:
            x = block.ln_1(x)
            x = x + block.attn(x)
            x = x + block.mlp(block.ln_2(x))            
        x = self.fmodel.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.fmodel.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.fmodel.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss 


    def get_lora_params_count(self,) -> int: 
        params = 0
        if self.wq_A is not None: params += sum([m.numel() for m in self.wq_A.parameters()])
        if self.wq_B is not None: params += sum([m.numel() for m in self.wq_B.parameters()])
        if self.wk_A is not None: params += sum([m.numel() for m in self.wk_A.parameters()])
        if self.wk_B is not None: params += sum([m.numel() for m in self.wk_B.parameters()])
        if self.wv_A is not None: params += sum([m.numel() for m in self.wv_A.parameters()])
        if self.wv_B is not None: params += sum([m.numel() for m in self.wv_B.parameters()])
        return params 