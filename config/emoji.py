# I/O
ckpt_name = 'emoji_gpt'  
eval_interval = 64 
eval_iters = 505 // 16
# data 
block_size = 128
batch_size = 16
gradient_accumulation_steps = 1
ds_path = './data/emojify-sft/ds'
# lora
adapts = set(['wq', 'wv', 'wk'])
rank = 4
alpha = 32
# adamw
max_iters = (2023 // 16) * 5
weight_decay = 0.0
# lr 
warmup_iters = 64
learning_rate = 5e-4
