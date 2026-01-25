"""
dft_utils.py
本文件包含实现 "维度自由Transformer" (Dimension-Free Transformer, DFT) 所需的核心数学工具。
所有函数都严格基于论文 "On Dimension-Free Transformer: An Application of STP to AI" by Daizhan Cheng
中定义的半张量积(STP)和相关理论。
"""
import math
import torch
def stp(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    计算两个矩阵A和B的半张量积 (Semi-Tensor Product, STP)。
    理论依据: 论文公式 (12), Definition 2.5 (i)。
    A (m, n) ⋉ B (p, q) = (A ⊗ I_{t/n}) @ (B ⊗ I_{t/p})，其中 t = lcm(n, p)。
    """
    m, n = A.shape
    p, q = B.shape
    t = math.lcm(n, p)
    t_div_n = t // n
    t_div_p = t // p
    eye_tn = torch.eye(t_div_n, device=A.device, dtype=A.dtype)
    eye_tp = torch.eye(t_div_p, device=B.device, dtype=B.dtype)
    A_kron = torch.kron(A, eye_tn)
    B_kron = torch.kron(B, eye_tp)
    result = torch.matmul(A_kron, B_kron)
    return result
def get_projection_matrix(dim_from: int, dim_to: int, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """
    严格按照论文公式 (22) 生成从 m 维空间到 n 维空间的投影矩阵 Π_n^m。
    Π_n^m = (n/t) * (I_n ⊗ 1_{t/n}^T) @ (I_m ⊗ 1_{t/m})，其中 t = lcm(m, n)。
    Args:
        dim_from (int): 源维度 (m)。
        dim_to (int): 目标维度 (n)。
    Returns:
        torch.Tensor: 形状为 (n, m) 的投影矩阵。
    """
    m = dim_from
    n = dim_to
    if m == n:
        return torch.eye(n, device=device, dtype=dtype)
    t = math.lcm(m, n)
    t_div_m = t // m
    t_div_n = t // n
    eye_n = torch.eye(n, device=device, dtype=dtype)
    eye_m = torch.eye(m, device=device, dtype=dtype)
    ones_tn = torch.ones((t_div_n, 1), device=device, dtype=dtype)
    ones_tm = torch.ones((t_div_m, 1), device=device, dtype=dtype)
    term1 = torch.kron(eye_n, ones_tn.T)
    term2 = torch.kron(eye_m, ones_tm)
    projection_matrix_unscaled = torch.matmul(term1, term2)
    projection_matrix = (n / t) * projection_matrix_unscaled
    return projection_matrix
def project(tensor: torch.Tensor, dim_to: int) -> torch.Tensor:
    """
    将一个 (序列长度, 嵌入维度) 的张量在序列长度维度上进行投影。
    这是维度自由操作的核心。
    """
    assert tensor.dim() == 2, f"Project function expects a 2D tensor (seq_len, n_embd), but got shape {tensor.shape}"
    dim_from = tensor.size(0)
    if dim_from == dim_to:
        return tensor
    proj_matrix = get_projection_matrix(dim_from, dim_to, device=tensor.device, dtype=tensor.dtype)
    return torch.matmul(proj_matrix, tensor)
def nominal_add(x: torch.Tensor, y: torch.Tensor, dim_target: int = None) -> torch.Tensor:
    """
    对两个张量进行名义加法 (Nominal Addition)。
    该函数现在可以处理两种情况：
    1. 如果两个张量维度相同，直接返回它们的和 (标准加法)。
    2. 如果维度不同，则将它们投影到目标维度再相加。
    """
    if x.shape == y.shape:
        return x + y
    dim_x = x.size(0) if x.dim() > 1 else len(x)
    dim_y = y.size(0) if y.dim() > 1 else len(y)
    if dim_target is None:
        dim_target = max(dim_x, dim_y)
    x_proj = project(x, dim_target)
    y_proj = project(y, dim_target)
    return x_proj + y_proj
def cross_dim_inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculates the dimension-free inner product of two VECTORS, x and y.
    Strictly follows the paper's Definition 2.8 (i) (formula 17).
    """
    assert x.dim() == 1, f"Input x must be a 1D vector, but got shape {x.shape}"
    assert y.dim() == 1, f"Input y must be a 1D vector, but got shape {y.shape}"
    dim_x = x.size(0)
    dim_y = y.size(0)
    if dim_x == dim_y:
        return torch.dot(x, y)
    t = math.lcm(dim_x, dim_y)
    x_matrix = x.view(-1, 1)
    y_matrix = y.view(-1, 1)
    x_proj = project(x_matrix, t)  
    y_proj = project(y_matrix, t)  
    x_proj_vec = x_proj.flatten()
    y_proj_vec = y_proj.flatten()
    inner_product = torch.dot(x_proj_vec, y_proj_vec) / t
    return inner_product
def generalized_linear_map(A: torch.Tensor, V_hypervector: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    实现广义线性映射 A ◊ V，用于注意力机制中值的加权聚合。
    理论依据: 论文公式 (96) 和 (97)。
    Args:
        A (torch.Tensor): 注意力矩阵，形状为 (s, t)。
        V_hypervector (list[torch.Tensor]): 值的超向量，一个包含 t 个张量的列表。
                                            每个张量 v_j 的形状是 (len_j, n_embd)。
    Returns:
        list[torch.Tensor]: 输出的超向量，一个包含 s 个张量的列表。
                            每个输出张量的长度是 V_hypervector 中最长的长度。
    """
    s = A.size(0)
    t = A.size(1)
    assert t == len(V_hypervector), "Attention matrix columns must match number of value vectors"
    if not V_hypervector:
        return []
    nominal_len = max(v.shape[0] for v in V_hypervector)
    n_embd = V_hypervector[0].shape[1]
    output_list = []
    for i in range(s):
        weighted_sum = torch.zeros(nominal_len, n_embd, device=A.device, dtype=A.dtype)
        for j in range(t):
            weight = A[i, j]
            value_vector = V_hypervector[j]
            weighted_v_proj = project(weight * value_vector, nominal_len)
            weighted_sum += weighted_v_proj
        output_list.append(weighted_sum)
    return output_list