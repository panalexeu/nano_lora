# the only non-sft ds
import tiktoken
import requests
from datasets import Dataset, DatasetDict

_URL = 'https://raw.githubusercontent.com/panalexeu/tinyshevchenko/refs/heads/main/tinyshevchenko.txt'

block_size = 256
test_size = 0.05
enc = tiktoken.get_encoding('gpt2')
path = './data/tiny-shevchenko/ds'

def chunk(tokens: list[int]) -> dict:
    # chunks of block_size + 1 tokens, since x and y are shifted by one.
    # consecutive chunks overlap by one token, so every token is a target exactly once
    ids = [tokens[i:i + block_size + 1] for i in range(0, len(tokens) - 1, block_size)]
    return {'ids': ids, 'instr_len': [0] * len(ids), 'len': [len(c) for c in ids]}

if __name__ == '__main__':
    content = requests.get(_URL).text
    tokens = enc.encode_ordinary(content)
    n_val = int(len(tokens) * test_size)

    ds = DatasetDict({
        'train': Dataset.from_dict(chunk(tokens[:-n_val])),
        'val': Dataset.from_dict(chunk(tokens[-n_val:])),
    })
    print(ds)
    ds.save_to_disk(path)