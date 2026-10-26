# DART config + emojify sft
# I/O
ckpt_name = 'emoji_gpt'  
eval_interval = 64 
eval_iters = 505 // 16
# data 
block_size = 128
batch_size = 16
gradient_accumulation_steps = 2
ds_path = './data/emojify-sft/ds'
# lora
adapts = set(['wq', 'wv'])
rank = 4
alpha = 32
# adamw
max_iters = (2023 // 32) * 5
weight_decay = 0.0
# lr 
warmup_iters = 16
learning_rate = 2e-4
