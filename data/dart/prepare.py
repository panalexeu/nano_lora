import tiktoken 
from datasets import load_dataset

_URL = "https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/"
_TRAINING_FILE = "dart-v1.1.1-full-train.json"
_DEV_FILE = "dart-v1.1.1-full-dev.json"
_TEST_FILE = "dart-v1.1.1-full-test.json"

num_proc = 8
block_size = 256
enc = tiktoken.get_encoding('gpt2')
path = './data/dart/ds'

TEMPLATE = "### Triples:\n{triples}\n\n### Text:\n"

def _linearize(tripleset): 
    return '\n'.join(f'{s} | {p} | {o}' for s,p,o in tripleset)

def process(batch): 
    out = {'ids': [], 'instr_len': [], 'len': []}
    for tripleset, annotations in zip(batch['tripleset'], batch['annotations']): 
        prompt_enc = enc.encode_ordinary(TEMPLATE.format(triples=_linearize(tripleset)))
        for a in annotations:
            ids = prompt_enc + enc.encode_ordinary(a['text']) + [enc.eot_token]
            out['ids'].append(ids)
            out['instr_len'].append(len(prompt_enc))
            out['len'].append(len(ids))
    return out

if __name__ == '__main__': 
    dataset = load_dataset('json', data_files={
        'train': _URL + _TRAINING_FILE,
        'val':   _URL + _DEV_FILE,
        'test':  _URL + _TEST_FILE,
    })    
    tokenized = dataset.map(process, batched=True, remove_columns=dataset['train'].column_names, num_proc=num_proc)

    # drop examples that don't fit in the context (x and y are len - 1 tokens)
    before = {split: len(ds) for split, ds in tokenized.items()}
    tokenized = tokenized.filter(lambda e: e['len'] <= block_size + 1, num_proc=num_proc)
    for split, ds in tokenized.items():
        print(f"{split}: dropped {before[split] - len(ds)} of {before[split]}")
    # for block size 256 drops only 17 examples:
    # train: dropped 13 of 62659
    # val: dropped 0 of 6980
    # test: dropped 4 of 12552
    tokenized.save_to_disk(path)
