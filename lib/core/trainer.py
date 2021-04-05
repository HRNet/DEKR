# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

from utils.utils import AverageMeter


def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter = AverageMeter()
    offset_loss_meter = AverageMeter()

    model.train()

    end = time.time()
    for i, (image, heatmap, mask, offset, offset_w) in enumerate(data_loader):
        data_time.update(time.time() - end)

        pheatmap, poffset = model(image)

        heatmap = heatmap.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        offset = offset.cuda(non_blocking=True)
        offset_w = offset_w.cuda(non_blocking=True)

        heatmap_loss, offset_loss = \
            loss_factory(pheatmap, poffset, heatmap, mask, offset, offset_w)

        loss = 0
        if heatmap_loss is not None:
            heatmap_loss_meter.update(heatmap_loss.item(), image.size(0))
            loss = loss + heatmap_loss
        if offset_loss is not None:
            offset_loss_meter.update(offset_loss.item(), image.size(0))
            loss = loss + offset_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{offset_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=image.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(
                          heatmap_loss_meter, 'heatmaps'),
                      offset_loss=_get_loss_info(offset_loss_meter, 'offset')
                  )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar(
                'train_heatmap_loss',
                heatmap_loss_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_offset_loss',
                offset_loss_meter.val,
                global_steps
            )
            writer_dict['train_global_steps'] = global_steps + 1


def _get_loss_info(meter, loss_name):
    msg = ''
    msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
        name=loss_name, meter=meter
    )

    return msg
