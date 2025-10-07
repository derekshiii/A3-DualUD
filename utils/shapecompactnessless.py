import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeCompactnessLoss(nn.Module):
    def __init__(self, lambda_val):
        super(ShapeCompactnessLoss, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, predicted_shape, target_shape):
        # 计算紧凑度度量
        compactness_metric = self.compute_compactness(target_shape)

        # 计算紧凑度度量损失
        compactness_loss = self.lambda_val * (1.0 - compactness_metric)

        # 组合损失
        combined_loss =  compactness_loss

        return combined_loss

    def compute_compactness(self, target_shape):
        # 计算目标形状的紧凑度
        area = torch.sum(target_shape)
        perimeter = torch.tensor(4.0 * target_shape.numel(), dtype=torch.float32)

        # 计算紧凑度度量
        compactness_metric = (4 * torch.pi * area) / (perimeter ** 2)

        return compactness_metric