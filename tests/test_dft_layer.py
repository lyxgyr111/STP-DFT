import torch
import torch.nn as nn


# ==============================================================================
# 1. 将你的核心DFT工具函数直接复制到这里
# ==============================================================================

def generalized_linear_map(A: torch.Tensor, V_hypervector: torch.Tensor) -> torch.Tensor:
    """
    【纯净实现】实现广义线性映射 Y = A ◊ V。
    此版本严格遵循 y_i = Σ(A_ij * v_j) 的定义，使用显式循环。
    """
    *batch_dims, m, n = A.shape
    *v_batch_dims, n_v, d = V_hypervector.shape

    assert batch_dims == list(v_batch_dims) and n == n_v, "批次维度和内积维度必须匹配"

    output_shape = (*batch_dims, m, d)
    Y_hypervector = torch.zeros(output_shape, device=A.device, dtype=A.dtype)

    # 为了简化，我们只处理一个批次维度 (nh) 和一个常规批次维度 (B)
    B, nh = batch_dims[0], batch_dims[1]

    for b in range(B):
        for h in range(nh):
            for i in range(m):
                y_i = torch.zeros(d, device=A.device, dtype=A.dtype)
                for j in range(n):
                    weight = A[b, h, i, j]
                    vector = V_hypervector[b, h, j, :]
                    y_i += weight * vector
                Y_hypervector[b, h, i, :] = y_i
    return Y_hypervector


def cross_dim_inner_product(Q_hypervector: torch.Tensor, K_hypervector: torch.Tensor) -> torch.Tensor:
    """
    【纯净实现】实现跨维度内积 S = Q ◎ K。
    """
    *q_batch_dims, m, d_q = Q_hypervector.shape
    *k_batch_dims, n, d_k = K_hypervector.shape

    assert q_batch_dims == list(k_batch_dims), "批次维度必须匹配"

    output_shape = (*q_batch_dims, m, n)
    S_matrix = torch.zeros(output_shape, device=Q_hypervector.device, dtype=Q_hypervector.dtype)

    B, nh = q_batch_dims[0], q_batch_dims[1]

    for b in range(B):
        for h in range(nh):
            for i in range(m):
                for j in range(n):
                    q_vec = Q_hypervector[b, h, i, :]
                    k_vec = K_hypervector[b, h, j, :]
                    # 这里是关键：我们只取两个向量中较短的长度进行点积
                    min_d = min(d_q, d_k)
                    dot_product = torch.dot(q_vec[:min_d], k_vec[:min_d])
                    S_matrix[b, h, i, j] = dot_product
    return S_matrix


# ==============================================================================
# 2. 定义一个极简的、独立的注意力层，不依赖任何外部文件
# ==============================================================================
class SimpleAttentionTestLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 这里什么都不需要，因为我们直接在forward里处理张量

    def forward(self, q, k, v):
        print("--> [Step 1] 开始进行跨维度内积 (Q ◎ K)...")
        # 注意：这里我们假设q, k, v已经是 (B, nh, T, d) 的形状
        att = cross_dim_inner_product(q, k)
        print("--> [Step 2] 跨维度内积完成。注意力矩阵形状:", att.shape)

        # 实际应用中会有softmax，但为了测试连通性，我们暂时跳过
        # att = torch.nn.functional.softmax(att, dim=-1)

        print("--> [Step 3] 开始进行广义线性映射 (A ◊ V)...")
        y = generalized_linear_map(att, v)
        print("--> [Step 4] 广义线性映射完成。输出张量形状:", y.shape)
        return y


# ==============================================================================
# 3. 主执行块：创建虚拟数据并运行一次前向传播
# ==============================================================================
if __name__ == '__main__':
    print("--- 开始决定性实验：单层注意力前向传播测试 ---")

    # 定义我们的维度不一致参数
    B = 2  # 批次大小
    nh = 4  # 注意力头数
    T_q = 16  # Q的序列长度
    T_k = 16  # K的序列长度
    T_v = 16  # V的序列长度 (必须等于T_k)

    d_q = 12  # Q的头部维度
    d_k = 10  # K的头部维度
    d_v = 8  # V的头部维度

    print("\n[参数设定]")
    print(f"  批次大小 (B): {B}, 注意力头数 (nh): {nh}")
    print(f"  序列长度 (T_q, T_k, T_v): {T_q, T_k, T_v}")
    print(f"  头部维度 (d_q, d_k, d_v): {d_q, d_k, d_v}  <-- 维度不一致!")

    # 创建虚拟的、维度不一致的输入张量
    # 形状: (Batch, Num_Heads, Sequence_Length, Head_Dim)
    q_tensor = torch.randn(B, nh, T_q, d_q)
    k_tensor = torch.randn(B, nh, T_k, d_k)
    v_tensor = torch.randn(B, nh, T_v, d_v)

    print("\n[输入张量形状]")
    print(f"  Q: {q_tensor.shape}")
    print(f"  K: {k_tensor.shape}")
    print(f"  V: {v_tensor.shape}")

    # 实例化我们的测试层
    attention_layer = SimpleAttentionTestLayer()

    try:
        print("\n--- [执行] 运行一次前向传播... ---")
        # 将张量移动到GPU（如果可用）
        if torch.cuda.is_available():
            print("  检测到CUDA，将张量移动到GPU...")
            q_tensor = q_tensor.cuda()
            k_tensor = k_tensor.cuda()
            v_tensor = v_tensor.cuda()
            attention_layer = attention_layer.cuda()

        # 执行核心测试
        output = attention_layer(q_tensor, k_tensor, v_tensor)

        print("\n--- [结果] ---")
        print("✅ ✅ ✅  实验成功！前向传播完成，没有发生错误。✅ ✅ ✅")
        print(f"最终输出张量的形状为: {output.shape}")

    except Exception as e:
        print("\n--- [结果] ---")
        print("❌ ❌ ❌ 实验失败！在前向传播过程中发生错误。❌ ❌ ❌")
        print("错误类型:", type(e).__name__)
        print("错误信息:", e)

