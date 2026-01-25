"""
This training script is STRICTLY for the Zero-Padding Control Group model.
It loads data into fixed-size padded tensors and calculates loss only on
valid (non-padded) tokens, providing a fair baseline for comparison.
MODIFIED TO COLLECT PERFORMANCE DATA.
"""
import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from dft_lingtianchong_model import GPTConfig, GPT_ZeroPadding as GPT
out_dir = 'out-dft-zeropad-baseline'
eval_interval = 250
log_interval = 10
eval_iters = 100
always_save_checkpoint = True
dataset = 'shakespeare_char_hetero'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
device = 'cpu'
compile = False
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
ddp = False
master_process = True
if ddp:
    pass
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ctx = nullcontext()
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
train_lengths = np.fromfile(os.path.join(data_dir, 'train_lengths.bin'), dtype=np.uint16)
train_offsets = np.concatenate(([0], np.cumsum(train_lengths[:-1]))).astype(np.uint64)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
val_lengths = np.fromfile(os.path.join(data_dir, 'val_lengths.bin'), dtype=np.uint16)
val_offsets = np.concatenate(([0], np.cumsum(val_lengths[:-1]))).astype(np.uint64)
def get_batch(split):
    tokens, lengths, offsets = (train_data, train_lengths, train_offsets) if split == 'train' else (
        val_data, val_lengths, val_offsets)
    ix = torch.randint(len(lengths), (batch_size,))
    x = torch.zeros(batch_size, block_size, dtype=torch.long)
    y = torch.full((batch_size, block_size), -100, dtype=torch.long)  
    for i, seq_idx in enumerate(ix):
        start, length = offsets[seq_idx], lengths[seq_idx]
        seq_len = min(length, block_size + 1)
        if seq_len > 1:
            seq_data = torch.from_numpy(tokens[start:start + seq_len - 1].astype(np.int64))
            x[i, :seq_len - 1] = seq_data
            y[i, :seq_len - 1] = torch.from_numpy(tokens[start + 1:start + seq_len].astype(np.int64))
    x, y = x.to(device), y.to(device)
    return x, y
iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=meta_vocab_size, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if master_process:
    print(f"--- Zero-Padding Control Group Experiment: {out_dir} ---")
    print(f"Model Parameters: {count_parameters(model) / 1e6:.3f}M")
    print("-------------------------------------------------------")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
def get_lr(it):
    if it < warmup_iters: return learning_rate * it / warmup_iters
    if it > lr_decay_iters: return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
X, Y = get_batch('train')
t0 = time.time()
t0_throughput = time.time()
total_tokens_processed = 0
throughput_log_interval = 100  
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
    for micro_step in range(gradient_accumulation_steps):
        batch_tokens = (Y != -100).sum().item()
        total_tokens_processed += batch_tokens
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')  
        loss.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
    if iter_num > 0 and (iter_num + 1) % throughput_log_interval == 0 and master_process:
        t1_throughput = time.time()
        elapsed_time = t1_throughput - t0_throughput
        tokens_per_second = total_tokens_processed / elapsed_time
        print("-" * 50)
        print(f"PERFORMANCE DATA @ iter {iter_num + 1}:")
        print(f"  - Time Elapsed: {elapsed_time:.2f}s")
        print(f"  - Total Tokens Processed in Window: {total_tokens_processed}")
        print(f"  - Throughput: {tokens_per_second:.2f} tokens/sec")
        print("-" * 50)
        t0_throughput = time.time()
        total_tokens_processed = 0
    iter_num += 1
    if iter_num > max_iters:
        break