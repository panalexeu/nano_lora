# LoRA: Low-Rank Adaptation of Large Language Models, table 11, DART config 
weight_decay = 0.0
batch_size = 4
gradient_accumulation_steps = 2
warmup_iters = 500
learning_rate = 2e-4
adapts = set(['wq', 'wv'])
rank = 4
alpha = 32
