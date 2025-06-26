# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import json
import math
import sys
from tqdm import tqdm
from typing import Iterable

import torch
import wandb

#import util.misc as misc
from . import lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    config=None):
    model.train(True)
    # metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # accum_iter = args.accum_iter

    running_loss = 0
    counts = 0
    image_log = {}

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (samples, label, descript) in tqdm(enumerate(data_loader)):
    # for data_iter_step, (samples, _) in tqdm(enumerate(data_loader)):

        # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
        # lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        lr_sched.adjust_learning_rate(optimizer, epoch, config)

        samples = samples.to(device, non_blocking=True)

        samples = samples[:, :3, :, :]
        # print(samples.shape)

        with torch.cuda.amp.autocast():
            loss, y, mask = model(samples, mask_ratio=config.model.mask_ratio)

        loss_value = loss.item()

        # running_loss += loss.detach().cpu().numpy()
        running_loss += loss_value
        counts += samples.shape[0]

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss /= accum_iter
        # loss_scaler(loss, optimizer, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
        
        # if (data_iter_step + 1) % accum_iter == 0:
        #     optimizer.zero_grad()
        optimizer.zero_grad()

        # torch.cuda.synchronize()

        # metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        # metric_logger.update(lr=lr)
        if config.wandb.wandb:
            if not image_log:
                statistics = json.load(open('/home/conradb/git/hephaestus-minicubes-ssl/statistics.json', "r"))
                phase_m = statistics['insar_difference']['mean']
                phase_std = statistics['insar_difference']['std']
                coh_m = statistics['insar_coherence']['mean']
                coh_std = statistics['insar_coherence']['std']
                dem_m = statistics['dem']['mean']
                dem_std = statistics['dem']['std']
                if label.any() == 1:
                    mask = mask.detach()
                    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)
                    mask = model.unpatchify(mask)
                    y = model.unpatchify(y)
                    im_masked = samples * (1 - mask)
                    im_paste = samples * (1 - mask) + y * mask

                    for i in range(label.shape[0]):
                        if label[i].item() == 1:
                            originals = [samples[i, 0, :, :]*phase_std + phase_m, samples[i, 1, :, :]*coh_std + coh_m, samples[i, 2, :, :]*dem_std + dem_m]
                            originals = [((sample - sample.min()) / (sample.max() - sample.min()))*255 for sample in originals]
                            originals = [wandb.Image(sample.unsqueeze(0)) for sample in originals]
                            masks = [im_masked[i, 0, :, :]*phase_std + phase_m, im_masked[i, 1, :, :]*coh_std + coh_m, im_masked[i, 2, :, :]*dem_std + dem_m]
                            masks = [((mask - mask.min()) / (mask.max() - mask.min()))*255 for mask in masks]
                            masks = [wandb.Image(mask.unsqueeze(0)) for mask in masks]
                            recons = [y[i, 0, :, :]*phase_std + phase_m, y[i, 1, :, :]*coh_std + coh_m, y[i, 2, :, :]*dem_std + dem_m]
                            recons = [((recon - recon.min()) / (recon.max() - recon.min()))*255 for recon in recons]
                            recons = [wandb.Image(recon.unsqueeze(0)) for recon in recons]
                            recons_vis = [im_paste[i, 0, :, :]*phase_std + phase_m, im_paste[i, 1, :, :]*coh_std + coh_m, im_paste[i, 2, :, :]*dem_std + dem_m]
                            recons_vis = [((recon_vis - recon_vis.min()) / (recon_vis.max() - recon_vis.min()))*255 for recon_vis in recons_vis]
                            recons_vis = [wandb.Image(recon_vis.unsqueeze(0)) for recon_vis in recons_vis]

                            outputs = {
                                'originals': originals,
                                'masked': masks,
                                'recon': recons,
                                'recon + vis': recons_vis,
                            }

                            image_log.update(outputs)
                            break
                    

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    epoch_loss = running_loss / counts
    eob_lr = lr # end of batch/epoch LR


    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return epoch_loss, eob_lr, image_log