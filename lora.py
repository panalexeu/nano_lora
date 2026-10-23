import os 
import math
import inspect
from typing import Literal
from dataclasses import dataclass, field, asdict

import torch 
import torch.nn as nn 
from torch.nn import functional as F

from model import GPT

@dataclass 
class LoRAConfig: 
    rank: int = 1
    alpha: int = 1
    model_type: Literal['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'] = 'gpt2'
    adapts: set[Literal['wq', 'wk', 'wv']] = field(default_factory=lambda: set(['wv']))
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
        self.scale = self.cfg.alpha / self.cfg.rank

        assert 1 <= self.cfg.rank <= self.embed 
        assert 1 <= len(self.cfg.adapts) <= 3

        # low rank matrices init 
        self.wq_A = self._init_A() if 'wq' in self.cfg.adapts else None
        self.wq_B = self._init_B() if 'wq' in self.cfg.adapts else None
        self.wk_A = self._init_A() if 'wk' in self.cfg.adapts else None
        self.wk_B = self._init_B() if 'wk' in self.cfg.adapts else None
        self.wv_A = self._init_A() if 'wv' in self.cfg.adapts else None
        self.wv_B = self._init_B() if 'wv' in self.cfg.adapts else None

    def _init_A(self) -> nn.ParameterList:
        params = nn.ParameterList([nn.Parameter(torch.empty(self.embed, self.cfg.rank)) for _ in range(self.fmodel.config.n_layer)])
        [nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.embed)) for p in params]
        return params

    def _init_B(self,) -> nn.ParameterList: 
        params = nn.ParameterList([nn.Parameter(torch.zeros(self.cfg.rank, self.embed)) for _ in range(self.fmodel.config.n_layer)])
        return params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.cfg.block_size, f"Cannot forward sequence of length {t}, block size is only {self.cfg.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.fmodel.transformer.wte(idx)
        pos_emb = self.fmodel.transformer.wpe(pos) 
        x = self.fmodel.transformer.drop(tok_emb + pos_emb)
        xq, xk, xv = None, None, None
        for i, block in enumerate(self.fmodel.transformer.h):
            h = block.ln_1(x)
            if 'wq' in self.cfg.adapts: xq = h @ self.wq_A[i] @ self.wq_B[i] * self.scale
            if 'wk' in self.cfg.adapts: xk = h @ self.wk_A[i] @ self.wk_B[i] * self.scale
            if 'wv' in self.cfg.adapts: xv = h @ self.wv_A[i] @ self.wv_B[i] * self.scale 
            x = x + block.attn(h, xq, xk, xv)
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


    def configure_optimizer(self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str): 
        params = [p for p in self.parameters() if p.requires_grad]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(params, weight_decay=weight_decay, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def merge_n_save(self, out_dir: str, ckpt_name: str): 
        if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True) 
        checkpoint = {
            'model': self.fmodel.state_dict(), 
            'model_args': asdict(self.fmodel.config), 
        }
        torch.save(checkpoint, os.path.join(out_dir, ckpt_name + '.pt'))
