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
    from dft_utils import stp, project, nominal_add, cross_dim_inner_product
except ImportError:
    print("无法导入dft_utils.py，请确保该文件存在。将使用占位符函数。")
    def project(x, dim_to):
        return x
    def nominal_add(x, y, dim_target):
        return x + y
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
    严格遵循论文 Definition 4.13 的维度自由线性层。
    这是一个三步过程：
    1. Project-Padding: 将每个序列从其原始长度 n_i 投影到名义长度 n_0。
    2. Standard Transformation: 对长度为 n_0 的序列应用标准线性变换。
    3. Project-Unpadding: 将结果从 n_0 投影回原始长度 n_i。
    """
    def __init__(self, in_features, out_features, bias=True, nominal_dim=256):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.nominal_dim = nominal_dim
    def forward(self, x_list: list[torch.Tensor]):
        original_dims = [x.size(0) for x in x_list]
        projected_list = [project(x, self.nominal_dim) for x in x_list]
        stacked_projected_x = torch.stack(projected_list)  
        stacked_transformed_x = self.linear(stacked_projected_x)  
        output_list = []
        for i in range(len(original_dims)):
            original_len = original_dims[i]
            transformed_tensor = stacked_transformed_x[i]
            unprojected_x = project(transformed_tensor, original_len)
            output_list.append(unprojected_x)
        return output_list
from dft_utils import cross_dim_inner_product, project, nominal_add
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
class CausalSelfAttentionDFT_Theoretical(nn.Module):
    """
    一个严格遵循论文理论的因果自注意力模块。
    该模块的核心特性是：
    1.  使用独立的、严格理论化的 DimensionFreeLinear_Theoretical 层来生成 Q, K, V，
        允许它们的维度 (d_q, d_k, d_v) 互不相同。
    2.  当 Q 和 K 的维度不同时，完全抛弃标准矩阵乘法 (`@`)，转而使用
        逐元素计算的“维度自由内积” (cross_dim_inner_product)，这正是
        论文中定义的 `Q ○ K` 算子。
    注意：为了理论的纯粹性，注意力分数的计算使用了Python循环，这将导致
    计算速度非常缓慢，仅适用于理论验证。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.is_asymmetric = config.qkv_dims is not None
        if self.is_asymmetric:
            print(f"Initializing STRICTLY THEORETICAL Asymmetric Attention with QKV dims: {config.qkv_dims}")
            self.q_dim, self.k_dim, self.v_dim = config.qkv_dims
            assert self.q_dim % self.n_head == 0, "q_dim must be divisible by n_head"
            assert self.k_dim % self.n_head == 0, "k_dim must be divisible by n_head"
            assert self.v_dim % self.n_head == 0, "v_dim must be divisible by n_head"
            self.q_proj = DimensionFreeLinear(config.n_embd, self.q_dim, bias=config.bias,
                                                          nominal_dim=config.block_size)
            self.k_proj = DimensionFreeLinear(config.n_embd, self.k_dim, bias=config.bias,
                                                          nominal_dim=config.block_size)
            self.v_proj = DimensionFreeLinear(config.n_embd, self.v_dim, bias=config.bias,
                                                          nominal_dim=config.block_size)
            self.c_proj = DimensionFreeLinear(self.v_dim, config.n_embd, bias=config.bias,
                                                          nominal_dim=config.block_size)
            self.q_head_size = self.q_dim // self.n_head
            self.k_head_size = self.k_dim // self.n_head
            self.v_head_size = self.v_dim // self.n_head
        else:  
            print("Initializing STRICTLY THEORETICAL Symmetric Attention.")
            assert config.n_embd % self.n_head == 0
            self.q_dim = self.k_dim = self.v_dim = config.n_embd
            self.q_head_size = self.k_head_size = self.v_head_size = config.n_embd // self.n_head
            self.c_attn = DimensionFreeLinear(config.n_embd, 3 * config.n_embd, bias=config.bias,
                                                          nominal_dim=config.block_size)
            self.c_proj = DimensionFreeLinear(config.n_embd, config.n_embd, bias=config.bias,
                                                          nominal_dim=config.block_size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    def forward(self, x_list: list[torch.Tensor]):
        if self.is_asymmetric:
            q_list = self.q_proj(x_list)
            k_list = self.k_proj(x_list)
            v_list = self.v_proj(x_list)
        else:
            qkv_list = self.c_attn(x_list)
            q_list, k_list, v_list = [], [], []
            for qkv in qkv_list:
                q, k, v = qkv.split(self.config.n_embd, dim=1)
                q_list.append(q);
                k_list.append(k);
                v_list.append(v)
        final_y_list = []
        for i in range(len(x_list)):
            q_full, k_full, v_full = q_list[i], k_list[i], v_list[i]
            T = q_full.size(0)
            q_heads = q_full.view(T, self.n_head, self.q_head_size)
            k_heads = k_full.view(T, self.n_head, self.k_head_size)
            v_heads = v_full.view(T, self.n_head, self.v_head_size)
            y_heads_list = []
            for h in range(self.n_head):
                q = q_heads[:, h, :]  
                k = k_heads[:, h, :]  
                v = v_heads[:, h, :]  
                att = torch.zeros(T, T, device=q.device, dtype=q.dtype)
                for row in range(T):
                    for col in range(T):
                        att[row, col] = cross_dim_inner_product(q[row], k[col])
                att = att / math.sqrt(self.q_head_size)
                causal_mask = torch.tril(torch.ones(T, T, device=q.device)).bool()
                att = att.masked_fill(~causal_mask, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  
                y_heads_list.append(y)
            y = torch.cat(y_heads_list, dim=-1)  
            final_y_list.append(y)
        y_list = self.c_proj(final_y_list)
        y_list = [self.resid_dropout(y) for y in y_list]
        return y_list
class MLP_DFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = DimensionFreeLinear(config.n_embd, 4 * config.n_embd, bias=config.bias, nominal_dim=config.block_size)
        self.gelu = nn.GELU()
        self.c_proj = DimensionFreeLinear(4 * config.n_embd, config.n_embd, bias=config.bias, nominal_dim=config.block_size)
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
        self.ln_1 = ListLayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionDFT_Theoretical(config)
        self.ln_2 = ListLayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP_DFT(config)
    def forward(self, x_list: list[torch.Tensor]):
        attn_out_list = self.attn(self.ln_1(x_list))
        x_list_after_attn = [x + attn_out for x, attn_out in zip(x_list, attn_out_list)]
        mlp_out_list = self.mlp(self.ln_2(x_list_after_attn))
        x_list_after_mlp = [x + mlp_out for x, mlp_out in zip(x_list_after_attn, mlp_out_list)]
        return x_list_after_mlp
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    head_dims:list[int]= None
    qkv_dims: list[int] = None
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
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, DimensionFreeLinear):
                torch.nn.init.normal_(module.linear.weight, mean=0.0, std=0.02)
                if module.linear.bias is not None:
                    torch.nn.init.zeros_(module.linear.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx_list: list[torch.Tensor], targets_list=None):
        """
        模型的前向传播，处理一批变长序列。
        `idx_list` 和 `targets_list` 都是Tensor的列表。
        """
        x_list = []
        for idx in idx_list:
            device = idx.device
            t = idx.size(0)
            assert t <= self.config.block_size, f"序列长度 {t} 超过了模型的最大长度 {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = nominal_add(tok_emb, pos_emb)
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
            logits = self.lm_head(x_list[0][-1, :].unsqueeze(0))
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
            logits, _ = self(idx_list)  
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_list[0] = torch.cat((idx_list[0], idx_next.squeeze(0)))
        return idx_list[0].unsqueeze(0)  