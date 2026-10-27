# LoRA: Low-Rank Adaptation of Large Language Models, table 11, DART config 
weight_decay = 0.0
gradient_accumulation_steps = 1
warmup_iters = 500
learning_rate = 2e-4
adapts = set(['wq', 'wv'])
rank = 4
alpha = 32
ds_path = './data/dart/ds'
block_size = 256 
batch_size = 8 
model_type = 'gpt2-medium'
max_iters = (62_659 // batch_size) * 5
eval_iters = 256
eval_interval = 1000