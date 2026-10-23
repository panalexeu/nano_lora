import tiktoken 
from datasets import load_dataset 

seed = 2004
num_proc = 8
test_size = 0.05 
enc = tiktoken.get_encoding('gpt2')
path = './data/alpaca-cleaned/ds'

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

    def process(exmpl: dict): 
        instr = exmpl['instruction'] + '\n'
        if input := exmpl['input']: 
            instr += input + '\n'

        instr_enc = enc.encode_ordinary(instr)
        output_enc = enc.encode_ordinary(exmpl['output'])
        ids = instr_enc + output_enc
        ids.append(enc.eot_token)

        return {'ids': ids, 'instr_len': len(instr_enc), 'len': len(ids)}

    tokenized = split_dataset.map(
        process, 
        remove_columns=['output', 'input', 'instruction'],
        desc='tokenizing the splits',
        num_proc=num_proc
    )
    # (Pdb) max(tokenized['train']['len'])
    # 2698
    # (Pdb) max(tokenized['val']['len'])
    # 997
    # (Pdb) 
    tokenized.save_to_disk(path)
