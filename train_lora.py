from contextlib import nullcontext

import torch
import tiktoken
import datasets 

from lora import LoRAGPT, LoRAConfig

# data
ds_path = './data/alpaca-cleaned/ds'
ds = datasets.load_from_disk(ds_path)
gradient_acumulation_steps = 5
batch_size = 5
block_size = 3072
# lora
rank = 1
alpha = 1 
model_type = 'gpt2'
adapts = set(['wv'])
# adamw optimizer
max_iters = 600000
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 
# learning rate decay settings
decay_lr = True
warmup_iters = 500 
lr = 2e-4
# system 
device = 'cuda'
dtype = 'bfloat16'
compile = True

# encoding 
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

def _get_batch(split):
    data = ds[split]
    idxs = torch.randint(len(data), (batch_size,)).tolist()
    rows = data[idxs]                       # dict of lists: rows['ids'], rows['instr_len'], ...

    xs, ys = [], []
    for ids, instr_len in zip(rows['ids'], rows['instr_len']):
        ids = ids[:block_size + 1]          # safety truncation (see note below)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        y[:instr_len - 1] = -1              # don't train on predicting the prompt
        xs.append(x)
        ys.append(y)

    x = torch.full((batch_size, block_size), enc.eot_token, dtype=torch.long)
    y = torch.full((batch_size, block_size), -1, dtype=torch.long)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        x[i, :len(xi)] = xi
        y[i, :len(yi)] = yi

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def _get_lr(it):
    # linear warmup from ~0 to max_lr
    if it < warmup_iters:                      
        return lr * (it + 1) / warmup_iters
    
    # linear decay from max_lr to 0
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    
    return lr * max(0.0, 1.0 - progress)

if __name__ == '__main__': 
    torch.manual_seed(2004)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    cfg = LoRAConfig(
        rank=rank,
        alpha=alpha,
        model_type=model_type, 
        adapts=adapts, 
        block_size=block_size
    )
    model = LoRAGPT(cfg)
    model.to(device)
    print('LoRA trainable params: ', model.get_lora_params_count())

    scaler = torch.amp.GradScaler(enabled=(dtype=='float16'))
    optimizer = model.configure_optimizer(weight_decay, lr, (beta1, beta2), device_type)
    if compile: 
        model = torch.compile(model)

    batch = _get_batch('train')

    while True: 
        pass

