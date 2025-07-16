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
from torcheval.metrics import MulticlassF1Score, MulticlassAUPRC, MulticlassAUROC

from timm.data import Mixup
from timm.utils import accuracy

from . import lr_sched
import utilities.utils as utils


class_dict = {0: "Non Deformation", 1: "Deformation"}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    config=None):

    model.train(True)

    f1 = MulticlassF1Score(num_classes=2, average=None).to(device) # f1 per class
    f1.reset()

    running_loss = 0
    counts = 0
    image_log = {}
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

        running_loss += loss.item()
        counts += 1

        preds = torch.argmax(outputs, dim=1)
        f1.update(preds, targets)
        

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

        if config.wandb.wandb:
            if not image_log:
                image_log = utils.log_image(image_log, samples, model, targets, y=outputs, mode='linprobe')

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    epoch_loss = running_loss / counts
    eob_lr = optimizer.param_groups[0]["lr"]
    f1_scores = f1.compute()

    log_dict = {'Epoch': epoch, 'train_loss': epoch_loss, 'end_of_epoch_lr': eob_lr}
    for idx in range(f1_scores.shape[0]):
            log_dict['Train ' + f1.__class__.__name__ + ' Class: ' + class_dict[idx]] = f1_scores[idx]

    print(f'Epoch {epoch}/{config.train.epochs}, Train Epoch Loss is {log_dict["train_loss"]}, end of epoch LR is {log_dict["end_of_epoch_lr"]}')
    print('Train F1 score (Non Deformation): ', f1_scores[0])
    print('Train F1 score (Deformation): ', f1_scores[1])

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return log_dict, image_log


@torch.no_grad()
def evaluate(data_loader, model, criterion, device, phase='Val', epoch=None):

    # metric_logger = misc.MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()

    running_loss = 0
    running_acc = 0
    counts = 0
    probs = []
    labels =  []

    metrics = utils.initialize_metrics()
    for metric in metrics:
        metric.reset()
        metric.to(device)

    # for batch in metric_logger.log_every(data_loader, 10, header):
    # for it, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
    for data_iter_step, (samples, targets, descript) in tqdm(enumerate(data_loader)):
        # images = batch[0]
        # target = batch[-1]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        samples = samples[:, :3, :, :]

        # compute output
        with torch.cuda.amp.autocast():
            output = model(samples)
            loss = criterion(output, targets)

        preds = torch.argmax(output, dim=1)

        for metric in metrics:
            if not isinstance(metric, (MulticlassAUPRC, MulticlassAUROC)):
                metric.update(preds, targets)
            else:
                metric.update(output, targets)

        running_loss += loss.item()
        counts += 1
        sftmx = torch.nn.Softmax()(output)
        probs.append(sftmx)
        labels.append(targets)

        acc1, = accuracy(output, targets, topk=(1,))

        running_acc += acc1.item()

    epoch_loss = running_loss / counts

    if phase == 'Val':
        log_dict = {'Epoch': epoch, 'val_loss': epoch_loss}
    
    metrics_vals = {}

    for metric in metrics:
        scores = metric.compute()
        metrics_vals[metric.__class__.__name__] = scores # test results dict
        if phase == 'Val':
            for idx in range(scores.shape[0]):
                log_dict['Val '+ metric.__class__.__name__ + ' Class: ' + class_dict[idx]] = scores[idx] # val results dict
        else:
            continue

    if phase == 'Val':
        print(f'Val Epoch Loss is {log_dict["val_loss"]}')
        print('Val Acc (Non Deformation): ', log_dict['Val '+ 'MulticlassAccuracy' + ' Class: ' + class_dict[0]])
        print('Val Acc (Deformation): ', log_dict['Val '+ 'MulticlassAccuracy' + ' Class: ' + class_dict[1]])
        print('Val F1 score (Non Deformation): ', log_dict['Val '+ 'MulticlassF1Score' + ' Class: ' + class_dict[0]])
        print('Val F1 score (Deformation): ', log_dict['Val '+ 'MulticlassF1Score' + ' Class: ' + class_dict[1]])



        

    #     batch_size = images.shape[0]
    #     metric_logger.update(loss=loss.item())
    #     metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if phase == 'Test':
        return metrics_vals
    else:
        return log_dict