"""
这是一个为生成出版级 (SCI Q2+) 注意力热力图而优化的独立脚本。
功能:
1. 加载已训练的零填充模型和投影填充模型。
2. 输入同一批可变长数据。
3. 生成两张独立的、高分辨率的、符合学术出版规范的PDF格式图表：
   - attention_zero_padding.pdf: 清晰展示“注意力浪费”现象。
   - attention_projection_padding.pdf: 清晰展示注意力的高效利用。
"""
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import matplotlib.pyplot as plt
import numpy as np
import pickle
from dataclasses import dataclass
import math
import torch.nn as nn
from torch.nn import functional as F
PROJ_MODEL_PATH = 'out-projpad-proposed/ckpt.pt'
ZERO_MODEL_PATH = 'out-zeropad-baseline/ckpt.pt'
META_PATH = 'data/shakespeare_char_hetero/meta.pkl'
DEVICE = 'cpu'
VIS_LAYER_IDX = -1  
VIS_HEAD_IDX = 0  
try:
    from dft_utils import project, nominal_add
except ImportError:
    print("警告: 无法导入 dft_utils.py。投影填充模型的可视化将失败。")
    def project(x, dim_to):
        raise NotImplementedError
    def nominal_add(x, y):
        raise NotImplementedError
@dataclass
class GPTConfigZero:
    block_size: int = 256;
    vocab_size: int = 65;
    n_layer: int = 4
    n_head: int = 4;
    n_embd: int = 128;
    dropout: float = 0.0;
    bias: bool = False
class CausalSelfAttention_ZeroPadding_Vis(nn.Module):
    def __init__(self, config):
        super().__init__();
        assert config.n_embd % config.n_head == 0;
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias);
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias);
        self.attn_dropout = nn.Dropout(config.dropout);
        self.resid_dropout = nn.Dropout(config.dropout);
        self.n_head = config.n_head;
        self.n_embd = config.n_embd;
        self.register_buffer("bias",
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                                               config.block_size))
    def forward(self, x, attention_mask=None, return_attention=False):
        B, T, C = x.size();
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2);
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2);
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2);
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2);
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)));
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'));
        if attention_mask is not None: att = att.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))
        att = F.softmax(att, dim=-1);
        final_attention_matrix = att;
        att = torch.nan_to_num(att);
        att = self.attn_dropout(att);
        y = att @ v;
        y = y.transpose(1, 2).contiguous().view(B, T, C);
        y = self.resid_dropout(self.c_proj(y));
        if return_attention: return y, final_attention_matrix
        return y
class MLP_ZeroPadding_Vis(nn.Module):
    def __init__(self, config): super().__init__(); self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd,
                                                                          bias=config.bias); self.gelu = nn.GELU(); self.c_proj = nn.Linear(
        4 * config.n_embd, config.n_embd, bias=config.bias); self.dropout = nn.Dropout(config.dropout)
    def forward(self, x): x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x); x = self.dropout(x); return x
class Block_ZeroPadding_Vis(nn.Module):
    def __init__(self, config): super().__init__(); self.ln_1 = nn.LayerNorm(config.n_embd,
                                                                             bias=config.bias); self.attn = CausalSelfAttention_ZeroPadding_Vis(
        config); self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias); self.mlp = MLP_ZeroPadding_Vis(config)
    def forward(self, x, attention_mask=None, return_attention=False):
        attn_out, attn_matrix = self.attn(self.ln_1(x), attention_mask=attention_mask, return_attention=True)
        x = x + attn_out;
        x = x + self.mlp(self.ln_2(x));
        if return_attention: return x, attn_matrix
        return x
class GPT_ZeroPadding_Vis(nn.Module):
    def __init__(self, config):
        super().__init__();
        self.config = config;
        self.transformer = nn.ModuleDict(
            dict(wte=nn.Embedding(config.vocab_size, config.n_embd), wpe=nn.Embedding(config.block_size, config.n_embd),
                 drop=nn.Dropout(config.dropout),
                 h=nn.ModuleList([Block_ZeroPadding_Vis(config) for _ in range(config.n_layer)]),
                 ln_f=nn.LayerNorm(config.n_embd, bias=config.bias)));
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False);
        self.transformer.wte.weight = self.lm_head.weight
    def forward(self, idx, targets=None, return_attention=False):
        device = idx.device;
        b, t = idx.size();
        pos = torch.arange(0, t, dtype=torch.long, device=device);
        attention_mask = (idx != 0).float();
        tok_emb = self.transformer.wte(idx);
        pos_emb = self.transformer.wpe(pos);
        x = self.transformer.drop(tok_emb + pos_emb);
        attention_maps = [block(x, attention_mask=attention_mask, return_attention=True)[1] for block in
                          self.transformer.h];
        x = self.transformer.ln_f(x);
        logits = self.lm_head(x);
        loss = None
        if targets is not None: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                                       ignore_index=-1)
        if return_attention: return logits, loss, attention_maps
        return logits, loss
@dataclass
class GPTConfigProj:
    block_size: int = 256;
    vocab_size: int = 65;
    n_layer: int = 4;
    n_head: int = 4;
    n_embd: int = 128;
    dropout: float = 0.0;
    bias: bool = True;
    head_dims: list[int] = None;
    qkv_dims: list[int] = None
class ListLayerNorm_Vis(nn.Module):
    def __init__(self, ndim, bias): super().__init__(); self.ln = nn.LayerNorm(ndim, bias=bias)
    def forward(self, input_list: list[torch.Tensor]): return [self.ln(x) for x in input_list]
class DimensionFreeLinear_Vis(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 nominal_dim=256): super().__init__(); self.linear = nn.Linear(in_features, out_features,
                                                                               bias=bias); self.nominal_dim = nominal_dim
    def forward(self, x_list: list[torch.Tensor]):
        original_dims = [x.size(0) for x in x_list];
        projected_list = [project(x, self.nominal_dim) for x in x_list];
        stacked_projected_x = torch.stack(projected_list);
        stacked_transformed_x = self.linear(stacked_projected_x);
        output_list = [project(stacked_transformed_x[i], original_dims[i]) for i in range(len(original_dims))];
        return output_list
class CausalSelfAttentionDFT_Vis(nn.Module):
    def __init__(self, config):
        super().__init__();
        self.config = config;
        self.n_head = config.n_head;
        assert config.n_embd % config.n_head == 0;
        self.head_size = config.n_embd // config.n_head;
        self.c_attn = DimensionFreeLinear_Vis(config.n_embd, 3 * config.n_embd, bias=config.bias,
                                              nominal_dim=config.block_size);
        self.c_proj = DimensionFreeLinear_Vis(config.n_embd, config.n_embd, bias=config.bias,
                                              nominal_dim=config.block_size);
        self.attn_dropout = nn.Dropout(config.dropout);
        self.resid_dropout = nn.Dropout(config.dropout)
    def forward(self, x_list: list[torch.Tensor], return_attention=False):
        qkv_list = self.c_attn(x_list);
        final_y_list = [];
        attention_matrix_for_vis = None
        for i, qkv in enumerate(qkv_list):
            T = qkv.size(0);
            q, k, v = qkv.split(self.config.n_embd, dim=1);
            q = q.view(T, self.n_head, self.head_size).transpose(0, 1);
            k = k.view(T, self.n_head, self.head_size).transpose(0, 1);
            v = v.view(T, self.n_head, self.head_size).transpose(0, 1);
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)));
            causal_mask = torch.tril(torch.ones(T, T, device=q.device)).bool();
            att = att.masked_fill(~causal_mask, float('-inf'));
            att = F.softmax(att, dim=-1)
            if i == 0 and return_attention: attention_matrix_for_vis = att
            att = self.attn_dropout(att);
            y = att @ v;
            y = y.transpose(1, 2).contiguous().view(T, self.config.n_embd);
            final_y_list.append(y)
        y_list = self.c_proj(final_y_list);
        y_list = [self.resid_dropout(y) for y in y_list]
        if return_attention: return y_list, attention_matrix_for_vis
        return y_list
class MLP_DFT_Vis(nn.Module):
    def __init__(self, config): super().__init__(); self.c_fc = DimensionFreeLinear_Vis(config.n_embd,
                                                                                        4 * config.n_embd,
                                                                                        bias=config.bias,
                                                                                        nominal_dim=config.block_size); self.gelu = nn.GELU(); self.c_proj = DimensionFreeLinear_Vis(
        4 * config.n_embd, config.n_embd, bias=config.bias, nominal_dim=config.block_size); self.dropout = nn.Dropout(
        config.dropout)
    def forward(self, x_list: list[torch.Tensor]): x_list = self.c_fc(x_list); x_list = [self.gelu(x) for x in
                                                                                         x_list]; x_list = self.c_proj(
        x_list); x_list = [self.dropout(x) for x in x_list]; return x_list
class BlockDFT_Vis(nn.Module):
    def __init__(self, config): super().__init__(); self.ln_1 = ListLayerNorm_Vis(config.n_embd,
                                                                                  bias=config.bias); self.attn = CausalSelfAttentionDFT_Vis(
        config); self.ln_2 = ListLayerNorm_Vis(config.n_embd, bias=config.bias); self.mlp = MLP_DFT_Vis(config)
    def forward(self, x_list: list[torch.Tensor], return_attention=False):
        attn_out_list, attn_matrix = self.attn(self.ln_1(x_list), return_attention=True)
        x_list_after_attn = [x + attn_out for x, attn_out in zip(x_list, attn_out_list)];
        mlp_out_list = self.mlp(self.ln_2(x_list_after_attn));
        x_list_after_mlp = [x + mlp_out for x, mlp_out in zip(x_list_after_attn, mlp_out_list)];
        if return_attention: return x_list_after_mlp, attn_matrix
        return x_list_after_mlp
class GPT_ProjectionPadding_Vis(nn.Module):
    def __init__(self, config):
        super().__init__();
        self.config = config;
        self.transformer = nn.ModuleDict(
            dict(wte=nn.Embedding(config.vocab_size, config.n_embd), wpe=nn.Embedding(config.block_size, config.n_embd),
                 drop=nn.Dropout(config.dropout),
                 h=nn.ModuleList([BlockDFT_Vis(config) for _ in range(config.n_layer)]),
                 ln_f=ListLayerNorm_Vis(config.n_embd, bias=config.bias)));
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False);
        self.transformer.wte.weight = self.lm_head.weight
    def forward(self, idx_list: list[torch.Tensor], targets_list=None, return_attention=False):
        x_list = [];
        for idx in idx_list: t = idx.size(0); pos = torch.arange(0, t, dtype=torch.long,
                                                                 device=idx.device); tok_emb = self.transformer.wte(
            idx); pos_emb = self.transformer.wpe(pos); x = nominal_add(tok_emb, pos_emb); x_list.append(
            self.transformer.drop(x))
        attention_maps = [block(x_list, return_attention=True)[1] for block in self.transformer.h];
        x_list = self.transformer.ln_f(x_list);
        all_logits = [self.lm_head(x) for x in x_list];
        loss = None
        if targets_list is not None: flat_logits = torch.cat(
            [l.view(-1, l.size(-1)) for l in all_logits]); flat_targets = torch.cat(
            [t.view(-1) for t in targets_list]); loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
        if return_attention: return all_logits, loss, attention_maps
        return all_logits, loss
def create_publication_plot(attention_matrix, title, filename, tokens, is_zero_padding=False, max_len=None):
    """
    生成并保存一张符合SCI Q2+标准的注意力热力图。
    """
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(6, 5))  
    im = ax.imshow(attention_matrix, cmap='viridis', aspect='equal')
    ax.set_xlabel("Key Position (Source Token)")
    ax.set_ylabel("Query Position (Target Token)")
    tick_step = max(1, len(tokens) // 10)
    ax.set_xticks(np.arange(0, len(tokens), tick_step))
    ax.set_yticks(np.arange(0, len(tokens), tick_step))
    ax.set_xticklabels(tokens[::tick_step], rotation=90)
    ax.set_yticklabels(tokens[::tick_step])
    if is_zero_padding and max_len:
        true_seq_len = len(tokens)
        ax.axvline(x=true_seq_len - 0.5, color='r', linestyle='--', linewidth=1.5)
        ax.axhline(y=true_seq_len - 0.5, color='r', linestyle='--', linewidth=1.5)
        ax.text(true_seq_len + 0.5, true_seq_len / 2, 'Padding Region\n(Wasted Attention)',
                color='red', fontsize=10, ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        ax.set_xlim(-0.5, max_len - 0.5)
        ax.set_ylim(max_len - 0.5, -0.5)  
    ax.set_title(title, pad=15)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("Attention Weight", rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  
    print(f"图表已成功保存至: {filename}")
if __name__ == '__main__':
    print("正在加载已训练的权重...")
    proj_ckpt = torch.load(PROJ_MODEL_PATH, map_location=DEVICE)
    proj_conf = GPTConfigProj(**proj_ckpt['model_args'])
    model_proj = GPT_ProjectionPadding_Vis(proj_conf)
    model_proj.load_state_dict(proj_ckpt['model'], strict=False)
    model_proj.eval().to(DEVICE)
    zero_ckpt = torch.load(ZERO_MODEL_PATH, map_location=DEVICE)
    zero_conf = GPTConfigZero(**zero_ckpt['model_args'])
    model_zero = GPT_ZeroPadding_Vis(zero_conf)
    model_zero.load_state_dict(zero_ckpt['model'], strict=False)
    model_zero.eval().to(DEVICE)
    print("权重加载完毕。")
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])
    sample_text = "O, she doth teach the torches to burn bright!"
    tokenized_sample = torch.tensor(encode(sample_text), dtype=torch.long, device=DEVICE)
    decoded_tokens = [itos.get(i.item(), '') for i in tokenized_sample]
    true_seq_len = len(tokenized_sample)
    print(f"将可视化样本 (长度 {true_seq_len}): '{sample_text}'")
    print("\n--- 正在处理零填充模型 ---")
    with torch.no_grad():
        block_size = zero_conf.block_size
        x_padded = torch.zeros(1, block_size, dtype=torch.long, device=DEVICE)
        x_padded[0, :true_seq_len] = tokenized_sample
        _, _, all_attentions_zero = model_zero(x_padded, return_attention=True)
        att_zero = all_attentions_zero[VIS_LAYER_IDX][0, VIS_HEAD_IDX, :, :].cpu().numpy()
    create_publication_plot(
        attention_matrix=att_zero,
        title="Attention Mechanism: Zero-Padding Baseline",
        filename="attention_zero_padding.pdf",
        tokens=decoded_tokens,
        is_zero_padding=True,
        max_len=block_size
    )
    print("\n--- 正在处理投影填充模型 ---")
    with torch.no_grad():
        _, _, all_attentions_proj = model_proj([tokenized_sample], return_attention=True)
        att_proj = all_attentions_proj[VIS_LAYER_IDX][VIS_HEAD_IDX].cpu().numpy()
    create_publication_plot(
        attention_matrix=att_proj,
        title="Attention Mechanism: Projection-Padding (Ours)",
        filename="attention_projection_padding.pdf",
        tokens=decoded_tokens
    )
    print("\n所有可视化任务完成。")