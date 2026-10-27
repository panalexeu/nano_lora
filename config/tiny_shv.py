# I/O
ckpt_name = 'shv_gpt'  
eval_interval = 64 
eval_iters = 112 // 8
# data 
block_size = 256
batch_size = 8
gradient_accumulation_steps = 1
ds_path = './data/tiny-shevchenko/ds'
# lora
adapts = set(['wq', 'wv', 'wk'])
rank = 4
alpha = 32
# adamw
max_iters = (2119 // 8) * 5
weight_decay = 0.0
# lr 
warmup_iters = 64
learning_rate = 5e-4
