import sys 
from contextlib import nullcontext

import torch 
import tiktoken
import torch.nn as nn 

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
max_new_tokens = 64 
seed = 2004
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True 
enc = tiktoken.get_encoding("gpt2")
stop_tokens = [enc.eot_token]
# -----------------------------------------------------------------------------

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

def _load_model(ckpt: dict) -> nn.Module: 
    gptconf = GPTConfig(**ckpt['model_args'])
    model = GPT(gptconf)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model

if __name__ == '__main__': 
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ckpt_path = sys.argv[1]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _load_model(ckpt)
    model.eval()
    model.to(device)
    model = torch.compile(model)
