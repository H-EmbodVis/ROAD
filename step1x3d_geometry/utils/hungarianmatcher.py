import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcherWithLoss(nn.Module):
    """
    这个类将匈牙利匹配器和损失计算整合在一起，
    处理[batch, tokens1, channel]和[batch, tokens2, channel]特征，
    使用余弦相似度构建成本矩阵，使用L1距离计算损失
    """

    def __init__(self, cost_cosine: float = 1.0):
        """
        创建匹配器和损失计算器

        参数:
            cost_cosine: 余弦相似度在匹配成本中的相对权重
        """
        super().__init__()
        self.matcher = HungarianMatcher(cost_l2=cost_cosine)

    def forward(self, features1, features2):
        """
        执行匹配并计算批量损失

        参数:
            features1: 第一个特征序列，形状为 [batch, tokens1, channel]
            features2: 第二个特征序列，形状为 [batch, tokens2, channel]

        返回:
            一个字典，包含:
                - 'loss': 标量张量，整个批次的平均损失
                - 'distances': 列表，每个元素是对应批次中匹配对的L1距离
                - 'indices': 匹配索引，用于后续分析
        """
        # 执行匈牙利匹配
        indices = self.matcher(features1, features2)

        #print(indices)

        # 计算匹配对的余弦距离
        distances = self.compute_matched_distance(features1, features2, indices)

        # 计算损失 (平均余弦距离)
        loss = self.compute_loss(distances)

        return loss

    def compute_matched_distance(self, features1, features2, indices):
        """计算匹配对的余弦距离"""
        batch_size = features1.shape[0]
        distances = []

        for b in range(batch_size):
            idx1, idx2 = indices[b]

            # 提取匹配的token
            feat1_matched = features1[b][idx1]  # [num_matches, channel]
            feat2_matched = features2[b][idx2]  # [num_matches, channel]

            # 计算L1距离
            dist = F.cosine_similarity(feat1_matched, feat2_matched,dim=-1).mean(dim=-1)
            distances.append(dist)

        return distances

    def compute_loss(self, distances):
        """
        计算损失:
        采用L1距离的平均值，因为我们希望相似的特征对具有更小的距离
        """
        # 收集所有批次的距离
        all_dists = torch.stack(distances)

        # 计算平均损失
        if all_dists.numel() == 0:
            dtype = distances[0].dtype if distances else torch.float32
            return torch.tensor(0.0, dtype=dtype, device=distances[0].device if distances else None)

        # 计算平均L1距离作为损失
        cos_loss = all_dists.mean()
        loss = 1 - cos_loss
        return loss


class HungarianMatcher(nn.Module):
    """内部使用的匈牙利匹配器，使用L2距离（欧氏距离）构建成本矩阵"""

    def __init__(self, cost_l2: float = 1.0):
        super().__init__()
        self.cost_l2 = cost_l2
        assert cost_l2 > 0, "L2距离的权重不能为0"

    @torch.no_grad()
    def forward(self, features1, features2):
        batch_size, tokens1, channels = features1.shape
        tokens2 = features2.shape[1]

        # 确保输入有效
        assert features1.shape[0] == features2.shape[0], "两个特征的批次大小必须一致"
        assert features1.shape[2] == features2.shape[2], "两个特征的通道数必须一致"
        assert tokens1 > 0 and tokens2 > 0, "特征序列不能包含零token"

        # --- 修改部分：计算 L2 距离矩阵 ---
        # 使用 torch.cdist 计算两组向量之间的欧氏距离 (batch, tokens1, tokens2)
        # p=2 代表 L2 范数
        dist_matrix = torch.cdist(features1, features2, p=2)

        # 成本直接等于 L2 距离
        cost_matrix = self.cost_l2 * dist_matrix

        # 执行匹配
        indices = []
        cost_matrix_cpu = cost_matrix.cpu()

        for b in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu[b])
            indices.append(
                (torch.as_tensor(row_ind, dtype=torch.int64, device=features1.device),
                 torch.as_tensor(col_ind, dtype=torch.int64, device=features1.device))
            )

        return indices