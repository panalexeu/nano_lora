import torch
from torch.nn import functional as F
import tiktoken

from lora import LoRAGPT, LoRAConfig

if __name__ == '__main__': 
    device = 'cpu'
    cfg = LoRAConfig(
        rank=1,
        model_type='gpt2', 
        adapts=set(['wk', 'wv', 'wq']), 
        block_size=1024
    )
    lora_gpt = LoRAGPT(cfg)
    lora_gpt.to(device)
    print('LoRA trainable params: ', lora_gpt.get_lora_params_count())

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    prompt = '<|endoftext|>'
    temperature = 1.0
    start_ids = encode(prompt)
    idx = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    for i in range(32): 
        lora_logits, _ = lora_gpt.forward(idx)
        fm_logits, _ = lora_gpt.fmodel.forward(idx)
        assert torch.allclose(lora_logits, fm_logits)
