import tiktoken
from datasets import load_dataset

seed = 2004
num_proc = 8 
block_size = 64
enc = tiktoken.get_encoding('gpt2')
path = './data/emojify-sft/ds'

def process(exmpl: dict): 
    text_enc = enc.encode_ordinary(exmpl['text'] + '\n\n')
    output_enc = enc.encode_ordinary(exmpl['emojification'])
    ids = text_enc + output_enc + [enc.eot_token]
    return {'ids': ids, 'instr_len': len(text_enc), 'len': len(ids)}

if __name__ == '__main__': 
    dataset = load_dataset('marioparreno/emojify-sft', num_proc=num_proc)
    dataset['val'] = dataset.pop('test')
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text', 'emojification'],
    #         num_rows: 2023
    #     })
    #     val: Dataset({
    #         features: ['text', 'emojification'],
    #         num_rows: 505
    #     })
    # })

    tokenized = dataset.map(
        process, 
        remove_columns=['text', 'emojification'], 
        desc='tokenizing the splits',
        num_proc=num_proc
    )

    # drop examples that don't fit in the context (x and y are len - 1 tokens)
    before = {split: len(ds) for split, ds in tokenized.items()}
    tokenized = tokenized.filter(lambda e: e['len'] <= block_size + 1, num_proc=num_proc)
    for split, ds in tokenized.items():
        print(f"{split}: dropped {before[split] - len(ds)} of {before[split]}")

    tokenized.save_to_disk(path)