# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss


class OffsetsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def smooth_l1_loss(self, pred, gt, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta)
        return loss

    def forward(self, pred, gt, weights):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(pred, gt) * weights
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss


class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_loss = HeatmapLoss() if cfg.LOSS.WITH_HEATMAPS_LOSS else None
        self.heatmap_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        self.offset_loss = OffsetsLoss() if cfg.LOSS.WITH_OFFSETS_LOSS else None
        self.offset_loss_factor = cfg.LOSS.OFFSETS_LOSS_FACTOR

    def forward(self, output, poffset, heatmap, mask, offset, offset_w):
        if self.heatmap_loss:
            heatmap_loss = self.heatmap_loss(output, heatmap, mask)
            heatmap_loss = heatmap_loss * self.heatmap_loss_factor
        else:
            heatmap_loss = None
        
        if self.offset_loss:
            offset_loss = self.offset_loss(poffset, offset, offset_w)
            offset_loss = offset_loss * self.offset_loss_factor
        else:
            offset_loss = None

        return heatmap_loss, offset_loss
