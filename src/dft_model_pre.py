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
    def forward(self, x_list: list[torch.Tensor]):
        output_list = []
        for x in x_list:
            projected_tokens = []
            for i in range(x.size(0)):
                token = x[i]
                projected_token = project(token, self.in_features)
                projected_tokens.append(projected_token)
            projected_x = torch.stack(projected_tokens)
            output = F.linear(projected_x, self.weight, self.bias)
            output_list.append(output)
        return output_list
class CausalSelfAttentionDFT(nn.Module):
    """
    无维度Transformer的因果自注意力机制。
    用基于STP的操作替换了标准的矩阵乘法。
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.c_attn = DimensionFreeLinear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = DimensionFreeLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
    def forward(self, x_list: list[torch.Tensor]):
        qkv_list = self.c_attn(x_list)
        final_output_list = []
        for qkv in qkv_list:
            T, C = qkv.size(0), qkv.size(1)
            q, k, v = qkv.split(self.n_embd, dim=1)
            q = q.view(T, self.n_head, self.head_size).transpose(0, 1)
            k = k.view(T, self.n_head, self.head_size).transpose(0, 1)
            v = v.view(T, self.n_head, self.head_size).transpose(0, 1)
            nh, T_q, hs = q.size()  
            T_k = k.size(1)
            att = torch.zeros((nh, T_q, T_k), device=q.device)
            for h in range(nh):
                for i in range(T_q):  
                    for j in range(T_k):  
                        if j <= i:
                            query_token = q[h, i, :]
                            key_token = k[h, j, :]
                            att[h, i, j] = cross_dim_inner_product(query_token, key_token)
                        else:
                            att[h, i, j] = float('-inf')
            att = att * (1.0 / math.sqrt(hs))
            att = F.softmax(att, dim=-1)
            y = generalized_linear_map(att, v)
            y = y.transpose(0, 1).contiguous().view(T, self.n_embd)
            final_output_list.append(y)
        y_list = self.c_proj(final_output_list)
        y_list = [self.resid_dropout(y) for y in y_list]
        return y_list
class MLP_DFT(nn.Module):
    """ 兼容无维度Transformer的前馈网络(MLP)模块。 """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = DimensionFreeLinear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU() 
        self.c_proj  = DimensionFreeLinear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x_list: list[torch.Tensor]):
        x_list = self.c_fc(x_list)
        x_list = [self.gelu(x) for x in x_list]
        x_list = self.c_proj(x_list)
        x_list = [self.dropout(x) for x in x_list]
        return x_list
class BlockDFT(nn.Module):
    """ 无维度Transformer的单个构建块 (Block)。 """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = ListLayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionDFT(config)
        self.ln_2 = ListLayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP_DFT(config)
    def forward(self, x_list: list[torch.Tensor]):
        attn_out_list = self.attn(self.ln_1(x_list))
        x_list_after_attn = []
        for x_seq, attn_out_seq in zip(x_list, attn_out_list):
            summed_tokens = []
            for i in range(x_seq.size(0)):
                summed_token = nominal_add(x_seq[i], attn_out_seq[i], self.config.n_embd)
                summed_tokens.append(summed_token)
            x_list_after_attn.append(torch.stack(summed_tokens))
        mlp_out_list = self.mlp(self.ln_2(x_list_after_attn))
        x_list_after_mlp = []
        for x_seq, mlp_out_seq in zip(x_list_after_attn, mlp_out_list):
            summed_tokens = []
            for i in range(x_seq.size(0)):
                summed_token = nominal_add(x_seq[i], mlp_out_seq[i], self.config.n_embd)
                summed_tokens.append(summed_token)
            x_list_after_mlp.append(torch.stack(summed_tokens))
        return x_list_after_mlp
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
            ln_f=ListLayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("模型参数量: %.2fM" % (self.get_num_params() / 1e6,))
    def get_num_params(self, non_embedding=True):
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
    def forward(self, idx_list: list[torch.Tensor], targets_list=None):
        """
        模型的前向传播，处理一批变长序列。
        `idx_list` 和 `targets_list` 都是Tensor的列表。
        """
        print(f"    [Model] Forward pass started for a batch of {len(idx_list)} sequences...")
        x_list = []
        for idx in idx_list:
            device = idx.device
            t = idx.size(0)
            assert t <= self.config.block_size, f"序列长度 {t} 超过了模型的最大长度 {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = torch.stack([nominal_add(tok_emb[i], pos_emb[i], self.config.n_embd) for i in range(t)])
            x_list.append(self.transformer.drop(x))
        for block in self.transformer.h:
            x_list = block(x_list)
        x_list = self.transformer.ln_f(x_list)
        if targets_list is not None:
            all_logits = []
            for x in x_list:
                logits = self.lm_head(x)
                all_logits.append(logits)
            flat_logits = torch.cat([l.view(-1, l.size(-1)) for l in all_logits])
            flat_targets = torch.cat([t.view(-1) for t in targets_list])
            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
        else:
            last_token_features = x_list[0][-1, :]  
            last_token_features_3d = last_token_features.unsqueeze(0).unsqueeze(0)  
            logits = self.lm_head(last_token_features_3d)  
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
        assert idx.dim() == 2 and idx.size(0) == 1, "Generation currently supports a single sequence (batch size 1)."
        idx_list = [idx.squeeze(0)] 
        for _ in range(max_new_tokens):
            if idx_list[0].size(0) > self.config.block_size:
                idx_list[0] = idx_list[0][-self.config.block_size:]
            logits, _ = self(idx_list) 
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_list[0] = torch.cat((idx_list[0], idx_next.squeeze(0)))
        return idx_list[0].unsqueeze(0) 