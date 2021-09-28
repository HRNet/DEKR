# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on deep-high-resolution-net.pytorch.
# (https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import csv
import os
import shutil
import time
import sys
sys.path.append("../lib")

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
import models

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}


CROWDPOSE_KEYPOINT_INDEXES = {
    0: 'left_shoulder',
    1: 'right_shoulder',
    2: 'left_elbow',
    3: 'right_elbow',
    4: 'left_wrist',
    5: 'right_wrist',
    6: 'left_hip',
    7: 'right_hip',
    8: 'left_knee',
    9: 'right_knee',
    10: 'left_ankle',
    11: 'right_ankle',
    12: 'head',
    13: 'neck'
}


def get_pose_estimation_prediction(cfg, model, image, vis_thre, transforms):
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    with torch.no_grad():
        heatmap_sum = 0
        poses = []

        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, scale
            )
        
        heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            return []
        else:
            if cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

            final_poses = get_final_preds(
                poses, center, scale_resized, base_size
            )

        final_results = []
        for i in range(len(scores)):
            if scores[i] > vis_thre:
                final_results.append(final_poses[i])

        if len(final_results) == 0:
            return []

    return final_results


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--visthre', type=float, default=0)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)
    csv_output_rows = []

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=False)
    else:
        raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps < args.inferenceFps:
        raise ValueError('desired inference fps is ' +
                         str(args.inferenceFps)+' but video fps is '+str(fps))
    skip_frame_cnt = round(fps / args.inferenceFps)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(skip_frame_cnt), (frame_width, frame_height))

    count = 0
    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()
        count += 1

        if not ret:
            break

        if count % skip_frame_cnt != 0:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_pose = image_rgb.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        now = time.time()
        pose_preds = get_pose_estimation_prediction(
            cfg, pose_model, image_pose, args.visthre, transforms=pose_transform)
        then = time.time()
        if len(pose_preds) == 0:
            count += 1
            continue

        print("Find person pose in: {} sec".format(then - now))

        new_csv_row = []
        for coords in pose_preds:
            # Draw each point on image
            for coord in coords:
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                new_csv_row.extend([x_coord, y_coord])

        total_then = time.time()
        text = "{:03.2f} sec".format(total_then - total_now)
        cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        csv_output_rows.append(new_csv_row)
        img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        cv2.imwrite(img_file, image_debug)
        outcap.write(image_debug)

    # write csv
    csv_headers = ['frame']
    if cfg.DATASET.DATASET_TEST == 'coco':
        for keypoint in COCO_KEYPOINT_INDEXES.values():
            csv_headers.extend([keypoint+'_x', keypoint+'_y'])
    elif cfg.DATASET.DATASET_TEST == 'crowd_pose':
        for keypoint in CROWDPOSE_KEYPOINT_INDEXES.values():
            csv_headers.extend([keypoint+'_x', keypoint+'_y'])
    else:
        raise ValueError('Please implement keypoint_index for new dataset: %s.' % cfg.DATASET.DATASET_TEST)

    csv_output_filename = os.path.join(args.outputDir, 'pose-data.csv')
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

    vidcap.release()
    outcap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
