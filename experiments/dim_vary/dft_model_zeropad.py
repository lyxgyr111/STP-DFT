"""
无维度Transformer模型 (基于nanoGPT)
本模型经过修改，以支持处理批次内变长的序列。
其核心思想源于 "On Dimension-Free Transformer" 论文，使用了基于半张量积(STP)的操作。
"""
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from dft_utils import stp, project, nominal_add, cross_dim_inner_product,generalized_linear_map
except ImportError:
    print("无法导入dft_utils.py，请确保该文件存在。将使用占位符函数。")
    def project(x, dim_to): return x
    def nominal_add(x, y, dim_target): return x + y
class ListLayerNorm(nn.Module):
    """
    一个能在Tensor列表上操作的LayerNorm。
    它会对列表中的每一个Tensor单独应用标准的LayerNorm。
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.ln = nn.LayerNorm(ndim, bias=bias)
    def forward(self, input_list: list[torch.Tensor]):
        return [self.ln(x) for x in input_list]
class DimensionFreeLinear(nn.Module):
    """
    一个可以通过投影处理不同维度输入的线性层。
    它在进行线性变换前，会先将输入投影到一个名义维度。
    这在概念上实现了论文中的 `A ◊ X` 操作。
    理论依据: 论文 Definition 4.13, 由公式 (82) 总结。
    为简单和高效起见，我们实现一个实践版本：
    1. 将每个token投影到一个名义维度。
    2. 执行标准的线性变换。
    输出将保持在目标维度空间中。
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self._init_weights()
    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
    def forward(self, x: torch.Tensor):  
        B, T, C = x.shape
        x_flat = x.view(B * T, C)
        projected_tokens = []
        for i in range(x_flat.size(0)):
            token = x_flat[i]
            projected_token = project(token, self.in_features)
            projected_tokens.append(projected_token)
        projected_x_flat = torch.stack(projected_tokens)  
        output_flat = F.linear(projected_x_flat, self.weight, self.bias)
        output = output_flat.view(B, T, self.out_features)
        return output
class CausalSelfAttentionDFT(nn.Module):
    """
    无维度Transformer的因果自注意力机制。
    用基于STP的操作替换了标准的矩阵乘法。
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_q_head = 32
        self.d_k_head = 8
        self.d_v_head = 16  
        print(f"INFO: CausalSelfAttentionDFT is in DIMENSION-VARYING mode.")
        print(f"      Head dimensions: Q={self.d_q_head}, K={self.d_k_head}, V={self.d_v_head}")
        self.d_q_total = self.n_head * self.d_q_head
        self.d_k_total = self.n_head * self.d_k_head
        self.d_v_total = self.n_head * self.d_v_head
        self.q_proj = DimensionFreeLinear(config.n_embd, self.d_q_total, bias=config.bias)
        self.k_proj = DimensionFreeLinear(config.n_embd, self.d_k_total, bias=config.bias)
        self.v_proj = DimensionFreeLinear(config.n_embd, self.d_v_total, bias=config.bias)
        self.c_proj = DimensionFreeLinear(self.d_v_total, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
    def forward(self, x: torch.Tensor):  
        """ 输入 x 的形状为 (B, T, C) """
        B, T, C = x.shape
        q = self.q_proj(x)  
        k = self.k_proj(x)  
        v = self.v_proj(x)  
        q = q.view(B, T, self.n_head, self.d_q_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_v_head).transpose(1, 2)
        att = torch.zeros(B, self.n_head, T, T, device=q.device, dtype=q.dtype)
        for b in range(B):
            for h in range(self.n_head):
                for i in range(T):
                    for j in range(T):
                        if j <= i:  
                            q_vec = q[b, h, i, :]
                            k_vec = k[b, h, j, :]
                            att[b, h, i, j] = cross_dim_inner_product(q_vec, k_vec)
                        else:
                            att[b, h, i, j] = float('-inf')
        att = F.softmax(att, dim=-1)
        self.last_attention_scores = att
        att = self.resid_dropout(att)
        y = torch.zeros(B, self.n_head, T, self.d_v_head, device=q.device, dtype=q.dtype)
        for b in range(B):
            for h in range(self.n_head):
                y[b, h, :, :] = generalized_linear_map(att[b, h, :, :], v[b, h, :, :])
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_v_total)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y  
class MLP_DFT(nn.Module):
    """ 兼容无维度Transformer的前馈网络(MLP)模块。 """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = DimensionFreeLinear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = DimensionFreeLinear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x: torch.Tensor): 
        """ 输入 x 的形状为 (B, T, C) """
        x = self.c_fc(x)
        x = self.gelu(x)    
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
class BlockDFT(nn.Module):
    """ 无维度Transformer的单个构建块 (Block)。 """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionDFT(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP_DFT(config)
    def forward(self, x: torch.Tensor):  
        """ 输入 x 的形状为 (B, T, C) """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([BlockDFT(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("模型参数量: %.2fM" % (self.get_num_params() / 1e6,))
    def get_num_params(self, non_embedding=True):
        """
        计算模型参数量。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, DimensionFreeLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):  
        """
        模型的前向传播，处理一批由外部补零的、固定长度的序列。
        `idx` 和 `targets` 都是标准张量。
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"序列长度 {t} 超过了模型的最大长度 {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)  
        pos_emb = self.transformer.wpe(pos)  
        x = tok_emb + pos_emb  
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        else:
            logits = self.lm_head(x[:, [-1], :])  
            loss = None
        return logits, loss
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("Loading pretrained GPT-2 weights into DFT model is not supported.")
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"带权重衰减的参数张量数量: {len(decay_params)}, 共 {num_decay_params:,} 个参数")
        print(f"不带权重衰减的参数张量数量: {len(nodecay_params)}, 共 {num_nodecay_params:,} 个参数")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"使用 fused AdamW: {use_fused}")
        return optimizer
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 估计模型浮点运算利用率 (MFU) """
        print("警告: MFU的估算在变长序列下不准确。")
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        为【实验组（补零填充）】模型设计的生成函数。
        它直接处理和传递标准张量。
        """
        assert idx.dim() == 2 and idx.size(0) == 1, "生成函数目前仅支持单一样本 (batch size 1)。"
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx