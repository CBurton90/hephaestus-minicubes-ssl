# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from tqdm import tqdm
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

from . import lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    config=None):
    model.train(True)
    running_loss = 0
    counts = 0
    # metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (samples, targets, descript) in tqdm(enumerate(data_loader)):

        # we use a per iteration (instead of per epoch) lr scheduler
        # NOTE - webdataset has no length so iteration per epoch (opposing the above)
        # if data_iter_step % accum_iter == 0:
        # lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        lr_sched.adjust_learning_rate(optimizer, epoch, config)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        samples = samples[:, :3, :, :]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        running_loss += loss.detach().cpu().numpy()
        counts += 1

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss /= accum_iter
        # loss_scaler(loss, optimizer, clip_grad=max_norm,
        #             parameters=model.parameters(), create_graph=False,
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False, update_grad=True)

        # if (data_iter_step + 1) % accum_iter == 0:
        #     optimizer.zero_grad()

        optimizer.zero_grad()

        # torch.cuda.synchronize()

        # metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        # metric_logger.update(lr=max_lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    epoch_loss = running_loss / counts

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return epoch_loss


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    # metric_logger = misc.MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()

    running_loss = 0
    running_acc = 0
    counts = 0
    probs = []
    labels =  []

    # for batch in metric_logger.log_every(data_loader, 10, header):
    for it, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        running_loss += loss.item()
        counts += 1
        sftmx = torch.nn.Softmax()(output)
        probs.append(sftmx)
        labels.append(target)

        acc1, = accuracy(output, target, topk=(1,))

        running_acc += acc1.item()

    #     batch_size = images.shape[0]
    #     metric_logger.update(loss=loss.item())
    #     metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return running_acc, running_loss, counts, torch.stack(probs).reshape(-1,2)[:,1], torch.stack(labels).reshape(-1)