"""
This training script is adapted for the Dimension-Free Transformer (DFT)
and its specific experimental validation plan.
... (script description remains the same) ...
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
from dft_model import GPTConfig, GPT
out_dir = 'out-dft-shakespeare'
eval_interval = 250
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'dft-shakespeare'
wandb_run_name = 'run' + str(time.time())
dataset = 'shakespeare_char_hetero'
data_loading_mode = 'variable'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
head_dims = None 
qkv_dims = None  
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
backend = 'nccl'
device = 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list, type(None)))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
if isinstance(head_dims, str):
    try:
        head_dims = eval(head_dims)
        print(f"Parsed head_dims from string to list: {head_dims}")
    except Exception as e:
        print(f"Error parsing head_dims string: {e}")
        head_dims = None
if isinstance(qkv_dims, str):
    try:
        qkv_dims = eval(qkv_dims)
        print(f"Parsed qkv_dims from string to list: {qkv_dims}")
    except Exception as e:
        print(f"Error parsing qkv_dims string: {e}")
        qkv_dims = None
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_local_rank = None
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
data_dir = os.path.join('data', dataset)
def load_hetero_data(split):
    tokens_path = os.path.join(data_dir, f'{split}.bin')
    lengths_path = os.path.join(data_dir, f'{split}_lengths.bin')
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode='r')
    lengths = np.fromfile(lengths_path, dtype=np.uint16)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.uint64)
    return {'tokens': tokens, 'lengths': lengths, 'offsets': offsets}
def load_fixed_data(split):
    return np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
if data_loading_mode == 'variable':
    train_data = load_hetero_data('train')
    val_data = load_hetero_data('val')
elif data_loading_mode == 'fixed':
    train_data = load_fixed_data('train')
    val_data = load_fixed_data('val')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    if data_loading_mode == 'variable':
        tokens, lengths, offsets = data['tokens'], data['lengths'], data['offsets']
        ix = torch.randint(len(lengths), (batch_size,))
        x_list, y_list = [], []
        for i in ix:
            start, length = offsets[i], lengths[i]
            if length > 1:
                end = start + length
                x = torch.from_numpy(tokens[start:end-1].astype(np.int64))
                y = torch.from_numpy(tokens[start+1:end].astype(np.int64))
                x_list.append(x)
                y_list.append(y)
        if not x_list: return get_batch(split)
    else: 
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x_list, y_list = [t for t in x], [t for t in y]
    x_list = [t.to(device) for t in x_list]
    y_list = [t.to(device) for t in y_list]
    return x_list, y_list
iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=meta_vocab_size, dropout=dropout,
                  head_dims=head_dims, qkv_dims=qkv_dims)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if master_process:
    print(f"--- Experiment: {out_dir} ---")
    print(f"Model Parameters: {count_parameters(model)/1e6:.3f}M")
    print("---------------------------------")
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
if compile:
    print("Compiling the model...")
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
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
local_iter_num = 0
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val'], "lr": lr})
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break
if ddp:
    destroy_process_group()