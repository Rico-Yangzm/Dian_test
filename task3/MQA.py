import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Query 独立投影，Key/Value 共享投影
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, self.d_head)
        self.Wv = nn.Linear(d_model, self.d_head)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 投影 Q/K/V
        Q = self.Wq(query).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(key).unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        V = self.Wv(value).unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)

        # 计算上下文向量
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.Wo(context), attn_weights


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_groups=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_groups == 0, "n_heads must be divisible by n_groups"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.d_head = d_model // n_heads
        self.heads_per_group = n_heads // n_groups

        # 每个组独立投影 Key/Value
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, self.d_head * n_groups)
        self.Wv = nn.Linear(d_model, self.d_head * n_groups)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 投影 Q
        Q = self.Wq(query).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 投影 K/V 并分组
        K = self.Wk(key).view(batch_size, seq_len, self.n_groups, self.d_head)
        K = K.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        V = self.Wv(value).view(batch_size, seq_len, self.n_groups, self.d_head)
        V = V.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)

        # 计算上下文向量
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.Wo(context), attn_weights


def visualize_attention(attn_weights, title="Attention Weights", group_structure=None):
    """
    可视化注意力权重
    参数:
        attn_weights: [batch_size, num_heads, query_len, key_len]
        title: 图表标题
        group_structure: GQA的分组结构 (可选)
    """
    # 取第一个样本的数据
    sample_attn = attn_weights[0].detach().cpu().numpy()
    num_heads, q_len, k_len = sample_attn.shape

    # 创建画布
    plt.figure(figsize=(16, 8))
    plt.suptitle(title, y=1.02, fontsize=14)

    # 根据分组结构调整布局
    if group_structure:
        groups, heads_per_group = group_structure
        rows = groups
        cols = heads_per_group + 1  # 最后一列显示组平均
    else:
        rows = 1
        cols = num_heads

    # 绘制每个注意力头
    for h in range(num_heads):
        ax = plt.subplot(rows, cols, h + 1)
        plt.imshow(sample_attn[h], cmap="viridis", vmin=0, vmax=1, aspect='auto')
        ax.set_title(f"Head {h + 1}", fontsize=8)

        # 添加分组标识
        if group_structure and (h + 1) % heads_per_group == 0:
            ax.text(1.1, 0.5, f'Group {(h // heads_per_group) + 1}',
                    rotation=270, va='center', transform=ax.transAxes)

    # 添加组平均可视化 (GQA专用)
    if group_structure:
        for g in range(groups):
            group_start = g * heads_per_group
            group_end = (g + 1) * heads_per_group
            group_avg = sample_attn[group_start:group_end].mean(axis=0)

            ax = plt.subplot(rows, cols, (g + 1) * cols)
            plt.imshow(group_avg, cmap="viridis", vmin=0, vmax=1, aspect='auto')
            ax.set_title(f"Group {g + 1} Avg", fontsize=8)

    # 添加公共颜色条
    plt.tight_layout()
    cax = plt.axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), cax=cax)
    plt.show()


# 测试代码 ------------------------------------------------------------

if __name__ == "__main__":
    # 参数配置
    d_model = 64  # 减小维度便于可视化
    seq_len = 8
    batch_size = 1


    # 构造测试数据（创建对角线注意力模式）
    def create_diagonal_input(seq_len, d_model):
        tensor = torch.zeros(1, seq_len, d_model)
        for i in range(seq_len):
            tensor[0, i, :] = i / seq_len
        return tensor


    query = key = value = create_diagonal_input(seq_len, d_model)

    # 测试 MQA
    print("Testing Multi-Query Attention:")
    mqa = MultiQueryAttention(d_model=d_model, n_heads=4)
    _, mqa_attn = mqa(query, key, value)
    visualize_attention(mqa_attn, "Multi-Query Attention")

    # 测试 GQA
    print("\nTesting Grouped-Query Attention:")
    gqa = GroupedQueryAttention(d_model=d_model, n_heads=8, n_groups=2)
    _, gqa_attn = gqa(query, key, value)
    visualize_attention(gqa_attn,
                        "Grouped-Query Attention",
                        group_structure=(2, 4))  # 2组，每组4个头

if __name__ == "__main__":
    # 参数配置
    d_model = 512
    n_heads = 8
    batch_size = 2
    seq_len = 10

    # 初始化输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = value = torch.randn(batch_size, seq_len, d_model)

    # 测试 MQA
    mqa = MultiQueryAttention(d_model, n_heads)
    mqa_output, mqa_attn = mqa(query, key, value)
    print(f"MQA 输出形状: {mqa_output.shape}")  # [2, 10, 512]
    print(f"MQA 注意力权重形状: {mqa_attn.shape}")  # [2, 8, 10, 10]

    # 测试 GQA（分4组）
    gqa = GroupedQueryAttention(d_model, n_heads, n_groups=4)
    gqa_output, gqa_attn = gqa(query, key, value)
    print(f"\nGQA 输出形状: {gqa_output.shape}")  # [2, 10, 512]
    print(f"GQA 注意力权重形状: {gqa_attn.shape}")  # [2, 8, 10, 10]


    # 参数数量对比
    def count_params(module):
        return sum(p.numel() for p in module.parameters())


    mha = nn.MultiheadAttention(d_model, n_heads)  # 标准多头注意力
    print("\n参数量对比:")
    print(f"MHA: {count_params(mha)}")  # 约 512*512*3 + 512*512 = 1,048,576
    print(f"MQA: {count_params(mqa)}")  # 约 512*512 + 512*64*2 = 327,680
    print(f"GQA: {count_params(gqa)}")  # 约 512*512 + 512*64*2*4 = 458,752
