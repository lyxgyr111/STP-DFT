"""
Post-hoc quantitative analysis for "attention waste" WITHOUT retraining.
It loads:
  - Zero-padding baseline checkpoint
  - Projection-padding (DFT) checkpoint (optional for attention, used as reference)
Then it samples variable-length sequences from val_heterogeneous.bin (or other detected .bin),
runs forward passes, extracts attention matrices, and computes:
1) PAMR-Key:   attention mass assigned to padding KEY columns  (usually ~0 with correct mask)
2) PAMR-Query: attention mass originating from padding QUERY rows (the "wasted area" in heatmap)
3) Entropy ratio: entropy(padding-query rows) / entropy(valid-query rows)   (weight-dependent)
Outputs (no training):
  - figures/pamr_query_scatter.pdf
  - figures/pamr_query_binned.pdf
  - figures/entropy_ratio_binned.pdf
  - analysis_outputs/pamr_table.csv
"""
import os
import glob
import math
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import sys
THIS_FILE = Path(__file__).resolve()
NANOGPT_ROOT = THIS_FILE.parents[1]   
sys.path.insert(0, str(NANOGPT_ROOT)) 
PROJ_MODEL_PATH = NANOGPT_ROOT / "out-projpad-proposed" / "ckpt.pt"
ZERO_MODEL_PATH = NANOGPT_ROOT / "out-zeropad-baseline" / "ckpt.pt"
META_PATH = NANOGPT_ROOT / "data" / "shakespeare_char_hetero" / "meta.pkl"
DEVICE = "cpu"  
VIS_LAYER_IDX = -1  
VIS_HEAD_IDX = None  
NUM_SAMPLES = 300
MIN_LEN = 16
SEED = 1337
try:
    from dft_utils import project, nominal_add
except ImportError:
    print("WARNING: cannot import dft_utils.py. Projection-padding model may fail to run.")
    def project(x, dim_to):
        raise NotImplementedError("project() not found. Please ensure dft_utils.py is importable.")
    def nominal_add(x, y):
        raise NotImplementedError("nominal_add() not found. Please ensure dft_utils.py is importable.")
@dataclass
class GPTConfigZero:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
class CausalSelfAttention_ZeroPadding_Vis(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
    def forward(self, x, attention_mask=None, return_attention=False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        if attention_mask is not None:
            att = att.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        final_attention_matrix = att  
        att = torch.nan_to_num(att)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        if return_attention:
            return y, final_attention_matrix
        return y
class MLP_ZeroPadding_Vis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
class Block_ZeroPadding_Vis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention_ZeroPadding_Vis(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP_ZeroPadding_Vis(config)
    def forward(self, x, attention_mask=None, return_attention=False):
        attn_out, attn_matrix = self.attn(
            self.ln_1(x), attention_mask=attention_mask, return_attention=True
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if return_attention:
            return x, attn_matrix
        return x
class GPT_ZeroPadding_Vis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block_ZeroPadding_Vis(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
    def forward(self, idx, targets=None, return_attention=False, attention_mask=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        if attention_mask is None:
            attention_mask = torch.ones(b, t, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        attention_maps = []
        for block in self.transformer.h:
            x, attn_matrix = block(x, attention_mask=attention_mask, return_attention=True)
            attention_maps.append(attn_matrix)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        if return_attention:
            return logits, loss, attention_maps
        return logits, loss
@dataclass
class GPTConfigProj:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    head_dims: list[int] = None
    qkv_dims: list[int] = None
class ListLayerNorm_Vis(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.ln = nn.LayerNorm(ndim, bias=bias)
    def forward(self, input_list: list[torch.Tensor]):
        return [self.ln(x) for x in input_list]
class DimensionFreeLinear_Vis(nn.Module):
    def __init__(self, in_features, out_features, bias=True, nominal_dim=256):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.nominal_dim = nominal_dim
    def forward(self, x_list: list[torch.Tensor]):
        original_dims = [x.size(0) for x in x_list]
        projected_list = [project(x, self.nominal_dim) for x in x_list]
        stacked_projected_x = torch.stack(projected_list)
        stacked_transformed_x = self.linear(stacked_projected_x)
        output_list = [project(stacked_transformed_x[i], original_dims[i]) for i in range(len(original_dims))]
        return output_list
class CausalSelfAttentionDFT_Vis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd // config.n_head
        self.c_attn = DimensionFreeLinear_Vis(
            config.n_embd, 3 * config.n_embd, bias=config.bias, nominal_dim=config.block_size
        )
        self.c_proj = DimensionFreeLinear_Vis(
            config.n_embd, config.n_embd, bias=config.bias, nominal_dim=config.block_size
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    def forward(self, x_list: list[torch.Tensor], return_attention=False):
        qkv_list = self.c_attn(x_list)
        final_y_list = []
        attention_matrix_for_vis = None
        for i, qkv in enumerate(qkv_list):
            T = qkv.size(0)
            q, k, v = qkv.split(self.config.n_embd, dim=1)
            q = q.view(T, self.n_head, self.head_size).transpose(0, 1)
            k = k.view(T, self.n_head, self.head_size).transpose(0, 1)
            v = v.view(T, self.n_head, self.head_size).transpose(0, 1)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
            causal_mask = torch.tril(torch.ones(T, T, device=q.device)).bool()
            att = att.masked_fill(~causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            if i == 0 and return_attention:
                attention_matrix_for_vis = att
            att = self.attn_dropout(att)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(T, self.config.n_embd)
            final_y_list.append(y)
        y_list = self.c_proj(final_y_list)
        y_list = [self.resid_dropout(y) for y in y_list]
        if return_attention:
            return y_list, attention_matrix_for_vis
        return y_list
class MLP_DFT_Vis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = DimensionFreeLinear_Vis(
            config.n_embd, 4 * config.n_embd, bias=config.bias, nominal_dim=config.block_size
        )
        self.gelu = nn.GELU()
        self.c_proj = DimensionFreeLinear_Vis(
            4 * config.n_embd, config.n_embd, bias=config.bias, nominal_dim=config.block_size
        )
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x_list: list[torch.Tensor]):
        x_list = self.c_fc(x_list)
        x_list = [self.gelu(x) for x in x_list]
        x_list = self.c_proj(x_list)
        x_list = [self.dropout(x) for x in x_list]
        return x_list
class BlockDFT_Vis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = ListLayerNorm_Vis(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionDFT_Vis(config)
        self.ln_2 = ListLayerNorm_Vis(config.n_embd, bias=config.bias)
        self.mlp = MLP_DFT_Vis(config)
    def forward(self, x_list: list[torch.Tensor], return_attention=False):
        attn_out_list, attn_matrix = self.attn(self.ln_1(x_list), return_attention=True)
        x_list_after_attn = [x + attn_out for x, attn_out in zip(x_list, attn_out_list)]
        mlp_out_list = self.mlp(self.ln_2(x_list_after_attn))
        x_list_after_mlp = [x + mlp_out for x, mlp_out in zip(x_list_after_attn, mlp_out_list)]
        if return_attention:
            return x_list_after_mlp, attn_matrix
        return x_list_after_mlp
class GPT_ProjectionPadding_Vis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([BlockDFT_Vis(config) for _ in range(config.n_layer)]),
                ln_f=ListLayerNorm_Vis(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
    def forward(self, idx_list: list[torch.Tensor], targets_list=None, return_attention=False):
        x_list = []
        for idx in idx_list:
            t = idx.size(0)
            pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = nominal_add(tok_emb, pos_emb)
            x_list.append(self.transformer.drop(x))
        attention_maps = []
        for block in self.transformer.h:
            x_list, attn_matrix = block(x_list, return_attention=True)
            attention_maps.append(attn_matrix)
        x_list = self.transformer.ln_f(x_list)
        all_logits = [self.lm_head(x) for x in x_list]
        loss = None
        if targets_list is not None:
            flat_logits = torch.cat([l.view(-1, l.size(-1)) for l in all_logits])
            flat_targets = torch.cat([t.view(-1) for t in targets_list])
            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
        if return_attention:
            return all_logits, loss, attention_maps
        return all_logits, loss
def find_val_bin():
    base = NANOGPT_ROOT / "data"
    candidates = [
        base / "shakespeare_char_hetero" / "val_heterogeneous.bin",
        base / "shakespeare_char_hetero" / "val.bin",
        base / "val_heterogeneous.bin",
        base / "val.bin",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    hits = []
    if base.exists():
        for p in base.rglob("*.bin"):
            if "val" in p.name.lower():
                hits.append(p)
    if hits:
        hits.sort()
        return str(hits[0])
    raise FileNotFoundError(
        f"Cannot find a validation .bin file under: {base}\n"
        f"Please check your heterogeneous dataset generation output."
    )
def sample_variable_length_sequence(bin_path, block_size, min_len=16, device="cpu"):
    """
    Randomly slice a contiguous segment from a token stream memmap.
    Returns (x_var, L) where x_var is 1D LongTensor length L.
    """
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    L = np.random.randint(min_len, block_size + 1)
    start = np.random.randint(0, len(data) - (L + 1))
    x = torch.from_numpy(data[start : start + L].astype(np.int64)).to(device)
    return x, int(L)
def _safe_entropy(p_row: np.ndarray) -> float:
    p = np.clip(p_row, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))
def extract_attn_matrix(attn_layer_tensor: torch.Tensor, head_idx=None) -> np.ndarray:
    """
    attn_layer_tensor for zero-padding model: [B, nh, T, T]
    Return [T, T] numpy: mean over heads if head_idx is None; else pick that head.
    """
    att = attn_layer_tensor[0]  
    if head_idx is None:
        att = att.mean(dim=0)   
    else:
        att = att[int(head_idx)]
    return att.detach().cpu().numpy()
def pamr_key(att_TT: np.ndarray, L: int) -> float:
    """
    Attention mass assigned to padding KEY columns (j>=L), normalized by total mass.
    With correct padding-key mask, this is typically ~0.
    """
    T = att_TT.shape[0]
    if L >= T:
        return 0.0
    pad_mass = float(np.sum(att_TT[:, L:T]))
    total_mass = float(np.sum(att_TT) + 1e-12)
    return pad_mass / total_mass
def pamr_query(att_TT: np.ndarray, L: int) -> float:
    """
    Attention mass originating from padding QUERY rows (i>=L), normalized by total mass.
    This corresponds to the "wasted" area (rows beyond true length) in your heatmap.
    """
    T = att_TT.shape[0]
    if L >= T:
        return 0.0
    waste_mass = float(np.sum(att_TT[L:T, :]))
    total_mass = float(np.sum(att_TT) + 1e-12)
    return waste_mass / total_mass
def entropy_ratio(att_TT: np.ndarray, L: int) -> float:
    """
    Compute:
      mean entropy of padding-query rows (i in [L,T)) / mean entropy of valid-query rows (i in [0,L))
    Entropy is computed over all columns (already masked for padding keys).
    This is weight-dependent and gives mechanistic evidence beyond pure geometry.
    """
    T = att_TT.shape[0]
    if L <= 1:
        return 1.0
    if L >= T:
        return 1.0
    ent_valid = []
    for i in range(0, L):
        ent_valid.append(_safe_entropy(att_TT[i, :]))
    ent_padq = []
    for i in range(L, T):
        ent_padq.append(_safe_entropy(att_TT[i, :]))
    mv = float(np.mean(ent_valid) + 1e-12)
    mp = float(np.mean(ent_padq) + 1e-12)
    return mp / mv
def make_length_bins(block_size: int):
    edges = [MIN_LEN, 32, 64, 96, 128, 160, 192, 224, block_size]
    edges = [e for e in edges if MIN_LEN <= e <= block_size]
    edges = sorted(list(dict.fromkeys(edges)))
    if edges[-1] != block_size:
        edges.append(block_size)
    if edges[0] != MIN_LEN:
        edges = [MIN_LEN] + edges
    if len(edges) < 2:
        edges = [MIN_LEN, block_size]
    return edges
def bin_index(L: int, edges: list[int]) -> int:
    for k in range(len(edges) - 1):
        if edges[k] <= L <= edges[k + 1]:
            return k
    return len(edges) - 2
def set_pub_style():
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        pass
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("analysis_outputs", exist_ok=True)
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta.get("vocab_size", None) or len(meta.get("itos", []))
        print(f"[meta] vocab_size = {vocab_size}")
    else:
        print("[meta] META_PATH not found, continue anyway.")
    print("Loading checkpoints...")
    zero_ckpt = torch.load(str(ZERO_MODEL_PATH), map_location=DEVICE)
    zero_conf = GPTConfigZero(**zero_ckpt["model_args"])
    model_zero = GPT_ZeroPadding_Vis(zero_conf)
    model_zero.load_state_dict(zero_ckpt["model"], strict=False)
    model_zero.eval().to(DEVICE)
    model_proj = None
    try:
        proj_ckpt = torch.load(str(PROJ_MODEL_PATH), map_location=DEVICE)
        proj_conf = GPTConfigProj(**proj_ckpt["model_args"])
        model_proj = GPT_ProjectionPadding_Vis(proj_conf)
        model_proj.load_state_dict(proj_ckpt["model"], strict=False)
        model_proj.eval().to(DEVICE)
        print("Projection model loaded (reference).")
    except Exception as e:
        print(f"WARNING: Projection model not loaded (will still run zero-padding analysis). Reason: {e}")
    block_size = int(zero_conf.block_size)
    print("block_size =", block_size)
    val_bin = find_val_bin()
    print("Using val bin:", val_bin)
    records = []
    print(f"Sampling {NUM_SAMPLES} variable-length sequences, running forward (no training)...")
    with torch.no_grad():
        for n in range(NUM_SAMPLES):
            x_var, L = sample_variable_length_sequence(
                val_bin, block_size=block_size, min_len=MIN_LEN, device=DEVICE
            )
            x_padded = torch.zeros(1, block_size, dtype=torch.long, device=DEVICE)
            x_padded[0, :L] = x_var
            attn_mask = torch.zeros(1, block_size, dtype=torch.float32, device=DEVICE)
            attn_mask[0, :L] = 1.0
            _, _, attn_maps_zero = model_zero(
                x_padded, return_attention=True, attention_mask=attn_mask
            )
            att_layer = attn_maps_zero[VIS_LAYER_IDX]  
            att_TT = extract_attn_matrix(att_layer, head_idx=VIS_HEAD_IDX)
            pk = pamr_key(att_TT, L)
            pq = pamr_query(att_TT, L)
            er = entropy_ratio(att_TT, L)
            theory_waste = 1.0 - (L / block_size) ** 2
            records.append({
                "L": L,
                "pamr_key": pk,
                "pamr_query": pq,
                "entropy_ratio": er,
                "theory_compute_waste": theory_waste,
            })
            if (n + 1) % 50 == 0:
                print(f"  {n+1}/{NUM_SAMPLES} done")
    raw_csv = os.path.join("analysis_outputs", "pamr_raw.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print("Saved:", raw_csv)
    edges = make_length_bins(block_size)
    nb = len(edges) - 1
    bins = [{"count": 0,
             "L_sum": 0.0,
             "pamr_key_sum": 0.0, "pamr_key_sq": 0.0,
             "pamr_query_sum": 0.0, "pamr_query_sq": 0.0,
             "entropy_ratio_sum": 0.0, "entropy_ratio_sq": 0.0,
             "theory_waste_sum": 0.0}
            for _ in range(nb)]
    for r in records:
        k = bin_index(int(r["L"]), edges)
        b = bins[k]
        b["count"] += 1
        b["L_sum"] += r["L"]
        b["pamr_key_sum"] += r["pamr_key"]
        b["pamr_key_sq"] += r["pamr_key"] ** 2
        b["pamr_query_sum"] += r["pamr_query"]
        b["pamr_query_sq"] += r["pamr_query"] ** 2
        b["entropy_ratio_sum"] += r["entropy_ratio"]
        b["entropy_ratio_sq"] += r["entropy_ratio"] ** 2
        b["theory_waste_sum"] += r["theory_compute_waste"]
    table_rows = []
    for k in range(nb):
        b = bins[k]
        if b["count"] == 0:
            continue
        c = b["count"]
        L_mean = b["L_sum"] / c
        def mean_std(sumv, sqv):
            m = sumv / c
            v = max(0.0, (sqv / c) - m * m)
            return m, math.sqrt(v)
        pk_m, pk_s = mean_std(b["pamr_key_sum"], b["pamr_key_sq"])
        pq_m, pq_s = mean_std(b["pamr_query_sum"], b["pamr_query_sq"])
        er_m, er_s = mean_std(b["entropy_ratio_sum"], b["entropy_ratio_sq"])
        tw_m = b["theory_waste_sum"] / c
        table_rows.append({
            "bin_range": f"[{edges[k]},{edges[k+1]}]",
            "count": c,
            "L_mean": round(L_mean, 4),
            "pamr_key_mean": round(pk_m, 6),
            "pamr_key_std": round(pk_s, 6),
            "pamr_query_mean": round(pq_m, 6),
            "pamr_query_std": round(pq_s, 6),
            "entropy_ratio_mean": round(er_m, 6),
            "entropy_ratio_std": round(er_s, 6),
            "theory_compute_waste_mean": round(tw_m, 6),
        })
    table_csv = os.path.join("analysis_outputs", "pamr_table.csv")
    with open(table_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)
    print("Saved:", table_csv)
    Ls = np.array([r["L"] for r in records], dtype=np.float32)
    pamr_q = np.array([r["pamr_query"] for r in records], dtype=np.float32)
    pamr_k = np.array([r["pamr_key"] for r in records], dtype=np.float32)
    ent_r = np.array([r["entropy_ratio"] for r in records], dtype=np.float32)
    theory_w = np.array([r["theory_compute_waste"] for r in records], dtype=np.float32)
    set_pub_style()
    plt.figure(figsize=(6.2, 4.4))
    plt.scatter(Ls, pamr_q, s=10, alpha=0.85, label="Zero-Padding (PAMR-Query)")
    plt.scatter(Ls, np.zeros_like(Ls), s=10, alpha=0.45, label="Projection-Padding (Ours, =0)")
    plt.xlabel("True sequence length L")
    plt.ylabel("PAMR-Query (mass from padding query rows)")
    plt.title("Quantifying Attention Waste from Padding Queries")
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join("figures", "pamr_query_scatter.pdf")
    plt.savefig(out1, bbox_inches="tight")
    plt.close()
    print("Saved:", out1)
    L_mean_arr = np.array([row["L_mean"] for row in table_rows], dtype=np.float32)
    pq_mean_arr = np.array([row["pamr_query_mean"] for row in table_rows], dtype=np.float32)
    pq_std_arr = np.array([row["pamr_query_std"] for row in table_rows], dtype=np.float32)
    tw_mean_arr = np.array([row["theory_compute_waste_mean"] for row in table_rows], dtype=np.float32)
    plt.figure(figsize=(6.2, 4.4))
    plt.plot(L_mean_arr, pq_mean_arr, marker="o", label="Zero-Padding (binned mean)")
    plt.fill_between(L_mean_arr, pq_mean_arr - pq_std_arr, pq_mean_arr + pq_std_arr, alpha=0.15)
    plt.plot(L_mean_arr, 1.0 - (L_mean_arr / block_size), linestyle="--", label="Geometry: (T-L)/T (query-row share)")
    plt.xlabel("True length L (binned mean)")
    plt.ylabel("PAMR-Query (mean ± std)")
    plt.title("Binned PAMR-Query vs Length")
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join("figures", "pamr_query_binned.pdf")
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print("Saved:", out2)
    er_mean_arr = np.array([row["entropy_ratio_mean"] for row in table_rows], dtype=np.float32)
    er_std_arr = np.array([row["entropy_ratio_std"] for row in table_rows], dtype=np.float32)
    plt.figure(figsize=(6.2, 4.4))
    plt.plot(L_mean_arr, er_mean_arr, marker="o", label="Zero-Padding (entropy ratio)")
    plt.fill_between(L_mean_arr, er_mean_arr - er_std_arr, er_mean_arr + er_std_arr, alpha=0.15)
    plt.axhline(1.0, linestyle="--", linewidth=1.0, label="=1.0 (same entropy)")
    plt.xlabel("True length L (binned mean)")
    plt.ylabel("Entropy(padding-query) / Entropy(valid-query)")
    plt.title("Mechanistic Evidence: Padding-Query Attention Entropy")
    plt.legend()
    plt.tight_layout()
    out3 = os.path.join("figures", "entropy_ratio_binned.pdf")
    plt.savefig(out3, bbox_inches="tight")
    plt.close()
    print("Saved:", out3)
    print("\nQuick summary (raw means):")
    print("  PAMR-Key   mean =", float(np.mean(pamr_k)))
    print("  PAMR-Query mean =", float(np.mean(pamr_q)))
    print("  EntropyRatio mean =", float(np.mean(ent_r)))
    print("\nAll done. Check ./figures and ./analysis_outputs")
if __name__ == "__main__":
    main()