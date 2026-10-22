import torch
from torch.nn import functional as F
import tiktoken

from lora import LoRAGPT, LoRAConfig

if __name__ == '__main__': 
    device = 'cpu'
    cfg = LoRAConfig(
        rank=1,
        model_type='gpt2', 
        adapts=set(['w_o']), 
        block_size=1024
    )
    lora_gpt = LoRAGPT(cfg)
    lora_gpt.to(device)

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    prompt = 'print("hello,'
    temperature = 1.0
    start_ids = encode(prompt)
    idx = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    for i in range(8): 
        logits, _ = lora_gpt.forward(idx)
        logits = logits[:, -1, :] / temperature   # [1, 50257]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)   # [1, 1]
        idx = torch.cat((idx, idx_next), dim=1)
        print(f'\r{decode(idx[0].tolist())}', end='')