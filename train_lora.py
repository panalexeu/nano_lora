# a lighter version of the original nanogpt training script. mainly the ddp functionality, float16 and the scaler were stripped.
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import tiktoken
import datasets 

from lora import LoRAGPT, LoRAConfig

# basically the hyperparams from table 11, DART (let's assume the DART task is the closest thing to instruction-following alpaca)
# tuned for the alpaca-cleaned ds, block_size 512
# I/O
out_dir = 'out'
ckpt_name = 'alpaca_gpt'
log_interval = 1
eval_interval = 2000
eval_iters = 2535 // 4 # divided by the actual batch size 
eval_only = False
# data
ds_path = './data/alpaca-cleaned/ds'
ds = datasets.load_from_disk(ds_path)
gradient_accumulation_steps = 2
batch_size = 4
block_size = 512
# lora
rank = 4
alpha = 32 
model_type = 'gpt2'
adapts = set(['wq', 'wv'])
# adamw optimizer
max_iters = (47_994 // 8) * 5  # divided by the effective batch size (batch_size * gradient_accumulation_steps) 
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0 
# learning rate decay settings
warmup_iters = 500 
learning_rate = 2e-4
# system 
device = 'cuda'
dtype = 'bfloat16'
compile = True
# encoding 
enc = tiktoken.get_encoding("gpt2")

def _get_batch(split):
    # if it works => do not touch! 
    data = ds[split]
    idxs = torch.randint(len(data), (batch_size,)).tolist()
    rows = data[idxs]                       # dict of lists: rows['ids'], rows['instr_len'], ...

    xs, ys = [], []
    for ids, instr_len in zip(rows['ids'], rows['instr_len']):
        ids = ids[:block_size + 1]          # safety truncation 
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

@torch.no_grad()
def _estimate_loss(model: nn.Module):
    out = {}
    model.eval()
    for split in ['train', 'val']: 
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = _get_batch(split)
            with ctx: 
                _, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

def _get_lr(it: int) -> float:
    # linear warmup from ~0 to max_lr
    if it < warmup_iters:                      
        return learning_rate * (it + 1) / warmup_iters
    # linear decay from max_lr to 0
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    
    return learning_rate * max(0.0, 1.0 - progress)

if __name__ == '__main__': 
    torch.manual_seed(2004)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
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

    optimizer = model.configure_optimizer(weight_decay, learning_rate, (beta1, beta2), device_type)
    if compile: 
        model = torch.compile(model)

    # training loop    
    iter_num = 0
    t0 = time.time()
    X, Y = _get_batch('train') 
    while True:
        # upd lr 
        lr = _get_lr(iter_num)
        for pg in optimizer.param_groups: 
            pg['lr'] = lr

        if iter_num % eval_interval == 0: 
            losses = _estimate_loss(model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") 
        if iter_num == 0 and eval_only: 
            break
        
        for micro_step in range(gradient_accumulation_steps): 
            with ctx:
                logits, loss = model(X,Y)
                loss = loss / gradient_accumulation_steps
            X,Y = _get_batch('train')
            loss.backward()

        if grad_clip != 0.0: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0: 
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps 
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        iter_num += 1
        if iter_num >= max_iters: 
            break 

    model.merge_n_save(out_dir, ckpt_name)
