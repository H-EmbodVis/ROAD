"""Minimal ULIP-2 PointBERT inference model.

Adapted from Salesforce/ULIP under the BSD 3-Clause license.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
from timm.models.layers import DropPath


@dataclass
class PointBERTConfig:
    trans_dim: int = 384
    depth: int = 18
    drop_path_rate: float = 0.1
    num_heads: int = 6
    group_size: int = 32
    num_group: int = 512
    encoder_dims: int = 256


def square_distance(source, target):
    distance = -2 * torch.matmul(source, target.transpose(1, 2))
    distance += source.square().sum(dim=-1).unsqueeze(2)
    distance += target.square().sum(dim=-1).unsqueeze(1)
    return distance


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, points):
        xyz, rgb = points[..., :3].contiguous(), points[..., 3:]
        indices = pointnet2_utils.furthest_point_sample(xyz, self.num_group)
        centers = pointnet2_utils.gather_operation(
            xyz.transpose(1, 2).contiguous(), indices
        ).transpose(1, 2).contiguous()
        neighbors = torch.topk(
            square_distance(centers, xyz),
            self.group_size,
            dim=-1,
            largest=False,
            sorted=False,
        ).indices
        batch_size, point_count, _ = points.shape
        flat = neighbors + torch.arange(batch_size, device=points.device).view(-1, 1, 1) * point_count
        local_xyz = xyz.reshape(batch_size * point_count, 3)[flat.reshape(-1)]
        local_rgb = rgb.reshape(batch_size * point_count, 3)[flat.reshape(-1)]
        local_xyz = local_xyz.reshape(batch_size, self.num_group, self.group_size, 3)
        local_rgb = local_rgb.reshape(batch_size, self.num_group, self.group_size, 3)
        return torch.cat([local_xyz - centers.unsqueeze(2), local_rgb], dim=-1), centers


class Encoder(nn.Module):
    def __init__(self, encoder_channel, input_dim=6):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, groups):
        batch_size, group_count, point_count, channels = groups.shape
        groups = groups.reshape(batch_size * group_count, point_count, channels)
        features = self.first_conv(groups.transpose(1, 2))
        global_feature = features.max(dim=2, keepdim=True).values
        features = self.second_conv(
            torch.cat([global_feature.expand(-1, -1, point_count), features], dim=1)
        ).max(dim=2).values
        return features.reshape(batch_size, group_count, -1)


class Mlp(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.fc1 = nn.Linear(dimension, dimension * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dimension * 4, dimension)
        self.drop = nn.Dropout(0.0)

    def forward(self, tokens):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(tokens)))))


class Attention(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.num_heads = heads
        self.scale = (dimension // heads) ** -0.5
        self.qkv = nn.Linear(dimension, dimension * 3, bias=False)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dimension, dimension)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, tokens):
        batch, count, channels = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, count, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = self.attn_drop((query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1))
        tokens = (attention @ value).transpose(1, 2).reshape(batch, count, channels)
        return self.proj_drop(self.proj(tokens))


class Block(nn.Module):
    def __init__(self, dimension, heads, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = Mlp(dimension)
        self.attn = Attention(dimension, heads)

    def forward(self, tokens):
        tokens = tokens + self.drop_path(self.attn(self.norm1(tokens)))
        return tokens + self.drop_path(self.mlp(self.norm2(tokens)))


class TransformerEncoder(nn.Module):
    def __init__(self, dimension, depth, heads, drop_paths):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dimension, heads, drop_paths[index]) for index in range(depth)]
        )

    def forward(self, tokens, positions):
        for block in self.blocks:
            tokens = block(tokens + positions)
        return tokens


class PointTransformerColored(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or PointBERTConfig()
        self.group_divider = Group(config.num_group, config.group_size)
        self.encoder = Encoder(config.encoder_dims)
        self.reduce_dim = nn.Linear(config.encoder_dims, config.trans_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, config.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, config.trans_dim)
        )
        drop_paths = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.blocks = TransformerEncoder(
            config.trans_dim, config.depth, config.num_heads, drop_paths
        )
        self.norm = nn.LayerNorm(config.trans_dim)

    def forward(self, points):
        neighborhoods, centers = self.group_divider(points)
        tokens = self.reduce_dim(self.encoder(neighborhoods))
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        positions = torch.cat(
            [self.cls_pos.expand(tokens.shape[0], -1, -1), self.pos_embed(centers)], dim=1
        )
        tokens = self.blocks(torch.cat([cls_token, tokens], dim=1), positions)
        tokens = self.norm(tokens)
        return torch.cat([tokens[:, 0], tokens[:, 1:].max(dim=1).values], dim=-1)


class ULIPPointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = PointTransformerColored()
        self.pc_projection = nn.Parameter(torch.empty(768, 1280))

    def forward(self, points):
        return self.point_encoder(points) @ self.pc_projection


def load_ulip_point_encoder(checkpoint_path):
    model = ULIPPointEncoder()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for name, value in state.items():
        name = name[len("module.") :] if name.startswith("module.") else name
        if name.startswith(("point_encoder.", "pc_projection")):
            cleaned[name] = value
    state = cleaned
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ULIP checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model
