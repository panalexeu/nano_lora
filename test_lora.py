# minimal tests to make sure that loragpt is fine and there are no problems in forwards/shapes.
# these tests make sense since initially lora initialized the b matrix as 0, that's why we can compare
# it with the frozen model and expect logits to match.
import os 
import shutil

import torch
import tiktoken

from lora import LoRAGPT, LoRAConfig

def test_get_lora_params_count(model: LoRAGPT): 
    assert model.get_lora_params_count() == 12 * 768 * 4 * 2 * 3 # for gpt2: n_layers * n_embd * rank * number of matrices (A,B) * number of adapters (wq, wk, wv)

def test_init_lora_adapters_match_fmodel(model: LoRAGPT, idx: torch.Tensor): 
    lora_logits, _ = model.forward(idx)
    fm_logits, _ = model.fmodel.forward(idx)
    assert torch.allclose(lora_logits, fm_logits)

def test_init_lora_adapters_merge_model_match_fmodel(model: LoRAGPT, idx: torch.Tensor): 
    fm_logits, _ = model.fmodel.forward(idx)
    merge_model = model.merge()
    merge_logits, _ = merge_model.forward(idx)
    assert torch.allclose(fm_logits, merge_logits)

def test_lora_merges_and_saves_ckpt(model: LoRAGPT): 
    model.merge_n_save('fake', 'fake')
    assert os.path.exists('./fake/fake.pt')
    shutil.rmtree('./fake')

if __name__ == '__main__': 
    device = 'cpu'
    rank = 4
    cfg = LoRAConfig(
        rank=rank, 
        model_type='gpt2', # the smallest one for tests 
        block_size=1024,
        adapts=set(['wk', 'wv', 'wq']), 
        alpha=1, 
    )
    lora_gpt = LoRAGPT(cfg)
    lora_gpt.to(device)

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    prompt = 'hello world!'
    start_ids = encode(prompt)
    idx = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    test_get_lora_params_count(lora_gpt)
    test_init_lora_adapters_match_fmodel(lora_gpt, idx)
    test_init_lora_adapters_merge_model_match_fmodel(lora_gpt, idx)
    test_lora_merges_and_saves_ckpt(lora_gpt)

    print('all tests are successful!')
