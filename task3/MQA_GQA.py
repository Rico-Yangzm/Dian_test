import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA)"""

    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 共享的Key/Value投影
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, self.d_head)
        self.Wv = nn.Linear(d_model, self.d_head)

        # KV Cache模拟
        self.register_buffer('k_cache', None)
        self.register_buffer('v_cache', None)

    def forward(self, query, key, value, use_cache=False):
        batch_size = query.size(0)

        # 投影操作
        Q = self.Wq(query).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(key).unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        V = self.Wv(value).unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # 更新KV Cache
        if use_cache:
            if self.k_cache is None:
                self.k_cache = K
                self.v_cache = V
            else:
                self.k_cache = torch.cat([self.k_cache, K], dim=2)
                self.v_cache = torch.cat([self.v_cache, V], dim=2)
            K, V = self.k_cache, self.v_cache

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(scores, dim=-1)
        return attn_weights


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA)"""

    def __init__(self, d_model=512, n_heads=8, n_groups=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        assert n_heads % n_groups == 0, "n_heads必须能被n_groups整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.d_head = d_model // n_heads
        self.heads_per_group = n_heads // n_groups

        # 修正投影维度
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, self.d_head * self.n_groups)  # 关键修正
        self.Wv = nn.Linear(d_model, self.d_head * self.n_groups)  # 关键修正

        # KV Cache模拟
        self.register_buffer('k_cache', None)
        self.register_buffer('v_cache', None)

    def forward(self, query, key, value, use_cache=False):
        batch_size, seq_len, _ = query.size()

        # 投影操作
        Q = self.Wq(query).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(key).view(batch_size, seq_len, self.n_groups, self.d_head)
        V = self.Wv(value).view(batch_size, seq_len, self.n_groups, self.d_head)

        # 广播到每组内的各个头
        K = K.unsqueeze(3)  # 添加heads_per_group维度
        K = K.expand(-1, -1, -1, self.heads_per_group, -1)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        V = V.unsqueeze(3)  # 添加heads_per_group维度
        V = V.expand(-1, -1, -1, self.heads_per_group, -1)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 更新KV Cache
        if use_cache:
            if self.k_cache is None:
                self.k_cache = K
                self.v_cache = V
            else:
                self.k_cache = torch.cat([self.k_cache, K], dim=2)
                self.v_cache = torch.cat([self.v_cache, V], dim=2)
            K, V = self.k_cache, self.v_cache

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(scores, dim=-1)
        return attn_weights


def visualize_attention(attn_weights, title, group_structure=None):
    """可视化注意力权重"""
    attn = attn_weights[0].detach().cpu().numpy()
    num_heads = attn.shape[0]

    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=14)

    if group_structure:
        groups, heads_per_group = group_structure
        for g in range(groups):
            plt.subplot(1, groups + 1, g + 1)
            group_attn = attn[g * heads_per_group:(g + 1) * heads_per_group].mean(0)
            plt.imshow(group_attn, cmap='viridis')
            plt.title(f'Group {g + 1} Avg')
    else:
        for h in range(num_heads):
            plt.subplot(1, num_heads, h + 1)
            plt.imshow(attn[h], cmap='viridis')
            plt.title(f'Head {h + 1}')

    plt.tight_layout()
    plt.show()


# 测试对比
if __name__ == "__main__":
    # 生成随机输入
    torch.manual_seed(42)
    seq_len = 10
    d_model = 64
    inputs = torch.randn(1, seq_len, d_model)  # [batch, seq, dim]

    # 初始化模块
    mqa = MultiQueryAttention(d_model=d_model, n_heads=8)
    gqa = GroupedQueryAttention(d_model=d_model, n_heads=8, n_groups=4)

    # ==== 关键修正：启用KV Cache ====
    # 第一次前向传播初始化Cache
    _ = mqa(inputs, inputs, inputs, use_cache=True)
    _ = gqa(inputs, inputs, inputs, use_cache=True)

    # 第二次前向传播使用Cache
    mqa_weights = mqa(inputs, inputs, inputs, use_cache=True)
    gqa_weights = gqa(inputs, inputs, inputs, use_cache=True)

    # 可视化对比
    print("MQA注意力模式：")
    visualize_attention(mqa_weights, "Multi-Query Attention")

    print("\nGQA注意力模式（每组2个头）：")
    visualize_attention(gqa_weights, "Grouped-Query Attention", group_structure=(4, 2))

    # KV Cache占用对比
    print("\nKV Cache内存占用对比：")
    print(f"MQA KV Cache尺寸: {mqa.k_cache.element_size() * mqa.k_cache.nelement() / 1024:.2f} KB")
    print(f"GQA KV Cache尺寸: {gqa.k_cache.element_size() * gqa.k_cache.nelement() / 1024:.2f} KB")
