# a nanogpt sample.py version that feels more like chatting with an llm experience 
import sys 
from typing import Literal
from contextlib import nullcontext

import torch 
import tiktoken
import torch.nn as nn 

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
max_new_tokens = 64 # 512
temperature = 0.8
top_k = 256 
seed = 2004
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True 
template: Literal['alpaca', 'emoji'] = 'emoji' # alpaca
enc = tiktoken.get_encoding("gpt2")
stop_tokens = [enc.eot_token]
# -----------------------------------------------------------------------------

prompt_templates = {
    'alpaca_instr': "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    'alpaca_input': "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    'emoji': '{instruction}\n\n'
}

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

def _form_prompt(instr: str, inp: str):
    if instr and inp and template == 'alpaca':
        return prompt_templates[template + '_input'].format(instruction=instr, input=inp)
    elif instr and not inp and template == 'alpaca':
        return prompt_templates[template + '_instr'].format(instruction=instr)
    elif instr and template == 'emoji': 
        return prompt_templates[template].format(instruction=instr) 

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

    while True: 
        # form prompt 
        instr = input('### Instruction: ')
        if not instr: continue # if enter was pressed just continue
        inp = input('### Input: ')
        prompt = _form_prompt(instr, inp) 

        # generate and stream! 
        start_ids = encode(prompt)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        printed = ''
        for idx in model.gen_iter(x, max_new_tokens, stop_tokens, temperature, top_k):
            text = decode(idx[0].tolist())
            if text.endswith('\ufffd'): continue # incomplete multi-byte character, wait for the next token
            print(text[len(printed):], end='', flush=True)
            printed = text
        print('\n')
