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

import argparse
import os
import sys
import stat
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from dataset import make_test_dataloader
from utils.utils import create_logger
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.rescore import rescore_valid

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, _ = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    data_loader, test_dataset = make_test_dataloader(cfg)
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    all_reg_preds = []
    all_reg_scores = []

    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
    for i, images in enumerate(data_loader):
        assert 1 == images.size(0), 'Test batch size should be 1'
        image = images[0].cpu().numpy()
        
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
                all_reg_preds.append([])
                all_reg_scores.append([])
            else:
                if cfg.TEST.MATCH_HMP:
                    poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

                final_poses = get_final_preds(
                    poses, center, scale_resized, base_size
                )
                if cfg.RESCORE.VALID:
                    scores = rescore_valid(cfg, final_poses, scores)
                all_reg_preds.append(final_poses)
                all_reg_scores.append(scores)

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()

    sv_all_preds = [all_reg_preds]
    sv_all_scores = [all_reg_scores]
    sv_all_name = [cfg.NAME]

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()

    for i in range(len(sv_all_preds)):
        print('Testing '+sv_all_name[i])
        preds = sv_all_preds[i]
        scores = sv_all_scores[i]
        if cfg.RESCORE.GET_DATA:
            test_dataset.evaluate(
                cfg, preds, scores, final_output_dir, sv_all_name[i]
            )
            print('Generating dataset for rescorenet successfully')
        else:
            name_values, _ = test_dataset.evaluate(
                cfg, preds, scores, final_output_dir, sv_all_name[i]
            )

            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(logger, name_value, cfg.MODEL.NAME)
            else:
                _print_name_value(logger, name_values, cfg.MODEL.NAME)


if __name__ == '__main__':
    main()
