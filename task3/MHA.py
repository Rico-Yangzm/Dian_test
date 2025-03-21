import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 线性变换层
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换 + 分头
        Q = self.Wq(query).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(key).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.Wv(value).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(scores, dim=-1)

        # 上下文向量
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.Wo(context), attn_weights


def visualize_attention(attn_weights, layer_name="Layer1"):
    """
    可视化注意力权重
    参数：
        attn_weights: [batch_size, num_heads, query_len, key_len]
        layer_name: 用于标题显示的层名称
    """
    # 取第一个样本的注意力权重
    sample_attn = attn_weights[0].detach().cpu().numpy()
    num_heads = sample_attn.shape[0]

    plt.figure(figsize=(15, 8))

    # 绘制每个注意力头的热力图
    for h in range(num_heads):
        plt.subplot(2, num_heads // 2, h + 1)
        plt.imshow(sample_attn[h], cmap="viridis", aspect='auto')
        plt.colorbar()
        plt.xticks(range(sample_attn.shape[-1]), [f"K{i + 1}" for i in range(sample_attn.shape[-1])])
        plt.yticks(range(sample_attn.shape[-2]), [f"Q{i + 1}" for i in range(sample_attn.shape[-2])])
        plt.title(f"Head {h + 1}")

    plt.suptitle(f"Attention Patterns - {layer_name}", y=1.02)
    plt.tight_layout()
    plt.show()


# 测试代码 ------------------------------------------------------------

if __name__ == "__main__":
    # 参数设置
    d_model = 64  # 减小维度便于可视化
    n_heads = 4
    seq_len = 8
    batch_size = 1

    # 创建模型
    mha = MultiHeadAttention(d_model, n_heads)

    # 生成可解释的测试数据
    query = key = value = torch.zeros(batch_size, seq_len, d_model)

    # 创建对角线关注模式（测试用）
    for i in range(seq_len):
        query[0, i, :] = i  # 每个位置有不同的查询值
        key[0, i, :] = i  # 键与查询相同
        value[0, i, :] = i  # 值也与查询相同

    # 前向计算
    output, attn_weights = mha(query, key, value)

    # 可视化展示
    print("注意力权重形状:", attn_weights.shape)
    visualize_attention(attn_weights)
