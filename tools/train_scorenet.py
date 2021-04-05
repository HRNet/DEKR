# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import _init_paths
import models
from config import cfg
from config import update_config
from utils.rescore import read_rescore_data
from utils.rescore import rescore_fit


def parse_args():
    parser = argparse.ArgumentParser(description='Train rescore network')
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


def main():
    args = parse_args()
    update_config(cfg, args)

    x_train, y_train = read_rescore_data(cfg)
    input_channels = x_train.shape[1]
    PredictOKSmodel = eval('models.'+'predictOKS'+'.get_pose_net')(
        cfg, input_channels, is_train=True
    )

    train_losses = rescore_fit(cfg, PredictOKSmodel, x_train, y_train)


if __name__ == '__main__':
    main()
