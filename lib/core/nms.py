# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def get_heat_value(pose_coord, heatmap):
    _, h, w = heatmap.shape
    heatmap_nocenter = heatmap[:-1].flatten(1,2).transpose(0,1)

    y_b = torch.clamp(torch.floor(pose_coord[:,:,1]), 0, h-1).long()
    x_l = torch.clamp(torch.floor(pose_coord[:,:,0]), 0, w-1).long()
    heatval = torch.gather(heatmap_nocenter, 0, y_b * w + x_l).unsqueeze(-1)
    return heatval


def cal_area_2_torch(v):
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h


def nms_core(cfg, pose_coord, heat_score):
    num_people, num_joints, _ = pose_coord.shape
    pose_area = cal_area_2_torch(pose_coord)[:,None].repeat(1,num_people*num_joints)
    pose_area = pose_area.reshape(num_people,num_people,num_joints)
    
    pose_diff = pose_coord[:, None, :, :] - pose_coord
    pose_diff.pow_(2)
    pose_dist = pose_diff.sum(3)
    pose_dist.sqrt_()
    pose_thre = cfg.TEST.NMS_THRE * torch.sqrt(pose_area)
    pose_dist = (pose_dist < pose_thre).sum(2)
    nms_pose = pose_dist > cfg.TEST.NMS_NUM_THRE

    ignored_pose_inds = []
    keep_pose_inds = []
    for i in range(nms_pose.shape[0]):
        if i in ignored_pose_inds:
            continue
        keep_inds = nms_pose[i].nonzero().cpu().numpy()
        keep_inds = [list(kind)[0] for kind in keep_inds]
        keep_scores = heat_score[keep_inds]
        ind = torch.argmax(keep_scores)
        keep_ind = keep_inds[ind]
        if keep_ind in ignored_pose_inds:
            continue
        keep_pose_inds += [keep_ind]
        ignored_pose_inds += list(set(keep_inds)-set(ignored_pose_inds))

    return keep_pose_inds


def pose_nms(cfg, heatmap_avg, poses):
    """
    NMS for the regressed poses results.

    Args:
        heatmap_avg (Tensor): Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """
    scale1_index = sorted(cfg.TEST.SCALE_FACTOR, reverse=True).index(1.0)
    pose_norm = poses[scale1_index]
    max_score = pose_norm[:,:,2].max() if pose_norm.shape[0] else 1

    for i, pose in enumerate(poses):
        if i != scale1_index:
            max_score_scale = pose[:,:,2].max() if pose.shape[0] else 1
            pose[:,:,2] = pose[:,:,2]/max_score_scale*max_score*cfg.TEST.DECREASE

    pose_score = torch.cat([pose[:,:,2:] for pose in poses], dim=0)
    pose_coord = torch.cat([pose[:,:,:2] for pose in poses], dim=0)

    if pose_coord.shape[0] == 0:
        return [], []

    num_people, num_joints, _ = pose_coord.shape
    heatval = get_heat_value(pose_coord, heatmap_avg[0])
    heat_score = (torch.sum(heatval, dim=1)/num_joints)[:,0]

    pose_score = pose_score*heatval
    poses = torch.cat([pose_coord.cpu(), pose_score.cpu()], dim=2)

    keep_pose_inds = nms_core(cfg, pose_coord, heat_score)
    poses = poses[keep_pose_inds]
    heat_score = heat_score[keep_pose_inds]
    
    if len(keep_pose_inds) > cfg.DATASET.MAX_NUM_PEOPLE:
        heat_score, topk_inds = torch.topk(heat_score,
                                            cfg.DATASET.MAX_NUM_PEOPLE)
        poses = poses[topk_inds]

    poses = [poses.numpy()]
    scores = [i[:, 2].mean() for i in poses[0]]

    return poses, scores
