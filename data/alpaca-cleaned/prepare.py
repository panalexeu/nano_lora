import tiktoken
from datasets import load_dataset

seed = 2004
num_proc = 8
test_size = 0.05
block_size = 1024
enc = tiktoken.get_encoding('gpt2')
path = './data/alpaca-cleaned/ds'

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

def format_prompt(instruction: str, inp: str = '') -> str:
    if inp:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
    return PROMPT_NO_INPUT.format(instruction=instruction)

def process(exmpl: dict):
    prompt = format_prompt(exmpl['instruction'], exmpl['input'])
    prompt_enc = enc.encode_ordinary(prompt)
    output_enc = enc.encode_ordinary(exmpl['output'])
    ids = prompt_enc + output_enc + [enc.eot_token]
    return {'ids': ids, 'instr_len': len(prompt_enc), 'len': len(ids)}

if __name__ == '__main__':
    dataset = load_dataset('yahma/alpaca-cleaned', num_proc=num_proc)  # instruction following dataset which is only 44.3MB in size!
    split_dataset = dataset['train'].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    # this results in:
    # DatasetDict({
    #     train: Dataset({
    #         features: ['output', 'input', 'instruction'],
    #         num_rows: 49172
    #     })
    #     val: Dataset({
    #         features: ['output', 'input', 'instruction'],
    #         num_rows: 2588
    #     })
    # })

    tokenized = split_dataset.map(
        process,
        remove_columns=['output', 'input', 'instruction'],
        desc='tokenizing the splits',
        num_proc=num_proc,
    )
    # (Pdb) max(tokenized['train']['len'])
    # 2741
    # (Pdb) max(tokenized['val']['len'])
    # 1026
    # (Pdb) 

    # drop examples that don't fit in the context (x and y are len - 1 tokens)
    before = {split: len(ds) for split, ds in tokenized.items()}
    tokenized = tokenized.filter(lambda e: e['len'] <= block_size + 1, num_proc=num_proc)
    for split, ds in tokenized.items():
        print(f"{split}: dropped {before[split] - len(ds)} of {before[split]}")
    tokenized.save_to_disk(path)