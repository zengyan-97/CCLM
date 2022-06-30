# Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training (https://arxiv.org/abs/2206.00621)
# Github: https://github.com/zengyan-97/CCLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import argparse
import os
import sys

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import math

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer

from models.model_pretrain_cclm import CrossViewLM

import utils
from dataset import create_dataset
from scheduler import create_scheduler
from optim import create_optimizer

from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hcopy
from accelerators.apex_ddp_accelerator import ApexDDPAccelerator


def reinit_scheduler_properties_mysched(optimizer: Optimizer, scheduler, cfg) -> None:
    """
    with ApexDDP, do re-init to avoid lr_scheduler warning.
    issue: https://github.com/pytorch/pytorch/issues/27595
    issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/841
    """
    args = cfg

    if scheduler.optimizer == optimizer:
        # from transformers import get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        scheduler.__init__(optimizer, lr_lambda, last_epoch=-1)


def image_multi_iter(model, image_batch, optimizer, accelerator, metric_logger, device):
    image, image_batch = image_batch[0].to(device, non_blocking=True), \
                         [t.to(device) if t is not None else None for t in image_batch[1:]]

    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
    text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2 = image_batch

    optimizer.zero_grad()

    loss = model(image, text_ids, text_atts, text_ids_masked=text_ids_masked,
                 masked_pos=masked_pos, masked_ids=masked_ids, text_ids_2=text_ids_2,
                 text_atts_2=text_atts_2, text_ids_masked_2=text_ids_masked_2,
                 masked_pos_2=masked_pos_2, masked_ids_2=masked_ids_2)

    loss_in_total = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm']
    if 'loss_ttc' in loss.keys():
        loss_in_total += loss['loss_ttc']
        loss_ttc = loss['loss_ttc'].item()
    else:
        loss_ttc = 0.0

    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_mm_img_itc=loss['loss_itc'].item())
    metric_logger.update(loss_mm_img_itm=loss['loss_itm'].item())
    metric_logger.update(loss_mm_img_mlm=loss['loss_mlm'].item())
    metric_logger.update(loss_mm_img_ttc=loss_ttc)


def region_multi_iter(model, region_batch, optimizer, accelerator, metric_logger, device):
    image, region_batch = region_batch[0].to(device, non_blocking=True), \
                          [t.to(device) if t is not None else None for t in region_batch[1:]]

    idx_to_group_img, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
    image_atts, target_bbox, is_image = region_batch

    if config['calc_image_bbox_loss']:
        is_image = None

    optimizer.zero_grad()

    loss = model(image, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos,
                 masked_ids=masked_ids,
                 image_atts=image_atts, idx_to_group_img=idx_to_group_img,
                 target_bbox=target_bbox, is_image=is_image, ret_bbox_loss=True)

    loss_in_total = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm'] + loss['loss_bbox'] + loss['loss_giou']

    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_mm_itc=loss['loss_itc'].item())
    metric_logger.update(loss_mm_itm=loss['loss_itm'].item())
    metric_logger.update(loss_mm_mlm=loss['loss_mlm'].item())
    metric_logger.update(loss_mm_bbox=loss['loss_bbox'].item())
    metric_logger.update(loss_mm_giou=loss['loss_giou'].item())


def image_mono_iter(model, mono_batch, optimizer, accelerator, metric_logger, device):
    # not debugged yet
    image, mono_batch = mono_batch[0].to(device, non_blocking=True), \
                        [t.to(device) if t is not None else None for t in mono_batch[1:]]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = mono_batch

    optimizer.zero_grad()

    loss = model(image, text_ids, text_atts, text_ids_masked=text_ids_masked,
                 masked_pos=masked_pos, masked_ids=masked_ids)

    loss_in_total = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm']
    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_itc=loss['loss_itc'].item())
    metric_logger.update(loss_itm=loss['loss_itm'].item())
    metric_logger.update(loss_mlm=loss['loss_mlm'].item())


def text_para_iter(model, batch, optimizer, accelerator, metric_logger, device):
    batch = [t.to(device) if t is not None else None for t in batch]

    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
    text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2 = batch

    optimizer.zero_grad()

    loss = model(None, text_ids=text_ids, text_atts=text_atts, text_ids_masked=text_ids_masked,
                 masked_pos=masked_pos, masked_ids=masked_ids, text_ids_2=text_ids_2,
                 text_atts_2=text_atts_2, text_ids_masked_2=text_ids_masked_2,
                 masked_pos_2=masked_pos_2, masked_ids_2=masked_ids_2)

    loss_in_total = loss['loss_ttc'] + loss['loss_ttm'] + loss['loss_mlm']
    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_tt_ttc=loss['loss_ttc'].item())
    metric_logger.update(loss_tt_ttm=loss['loss_ttm'].item())
    metric_logger.update(loss_tt_mlm=loss['loss_mlm'].item())


def text_para_iter_tlm(model, batch, optimizer, accelerator, metric_logger, device):
    batch = [t.to(device) if t is not None else None for t in batch]

    text_ids, text_atts, text_ids_2, text_atts_2, \
    text_ids_masked, text_atts_masked, masked_pos, masked_ids = batch

    optimizer.zero_grad()

    loss = model(None, text_ids=text_ids, text_atts=text_atts,
                 text_ids_2=text_ids_2, text_atts_2=text_atts_2,
                 text_ids_masked=text_ids_masked, text_atts_masked=text_atts_masked,
                 masked_pos=masked_pos, masked_ids=masked_ids)

    loss_in_total = loss['loss_ttc'] + loss['loss_mlm']
    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    if accelerator_clip_grad_norm > 0:
        accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    optimizer.step()

    metric_logger.update(loss_tt_ttc=loss['loss_ttc'].item())
    metric_logger.update(loss_tt_ttm=0.0)
    metric_logger.update(loss_tt_mlm=loss['loss_mlm'].item())


def train(model, general_loader, region_loader, mono_loader, text_loader, optimizer, epoch_info, device, scheduler, config, accelerator, checkpointer):
    model.train()
    start_epoch, _ = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")

    # multilingual images
    metric_logger.add_meter('loss_mm_img_itc', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
    metric_logger.add_meter('loss_mm_img_itm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
    metric_logger.add_meter('loss_mm_img_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
    metric_logger.add_meter('loss_mm_img_ttc', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_large', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train step: [{}]'.format(start_epoch)
    assert start_epoch == 0
    print_freq = 50

    world_size = utils.get_world_size()
    step_per_epoch = math.ceil(config['train_dataset_size']/(config['batch_size']*world_size))
    assert step_per_epoch > 1
    global_step = 0  # start from 0

    # image_iter = iter(general_loader)

    # optional
    if region_loader is not None:
        # multilingual regions
        region_iter = iter(region_loader)
        metric_logger.add_meter('loss_mm_itc', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_mm_itm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_mm_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_mm_bbox', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_mm_giou', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))

    else:
        region_iter = None

    if mono_loader is not None:
        # monolingual images
        mono_iter = iter(mono_loader)
        metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))

    else:
        mono_iter = None

    if text_loader is not None:
        # parallel texts
        text_iter = iter(text_loader)
        metric_logger.add_meter('loss_tt_ttc', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_tt_ttm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
        metric_logger.add_meter('loss_tt_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.2f}'))
    else:
        text_iter = None

    for i, image_batch in enumerate(metric_logger.log_every(general_loader, print_freq, header, step_per_epoch, epoch_info)):
        if (text_iter is not None) and (random.random() < config['texts_para']['iter_perc']):
            if 'use_tlm' in config and config['use_tlm']:
                text_para_iter_tlm(model, next(text_iter), optimizer, accelerator, metric_logger, device)
            else:
                text_para_iter(model, next(text_iter), optimizer, accelerator, metric_logger, device)

        if (mono_iter is not None) and (random.random() < config['images_mono']['iter_perc']):
            image_mono_iter(model, next(mono_iter), optimizer, accelerator, metric_logger, device)

        ### main iter
        if region_iter is not None:
            if global_step == 0:  # hack
                region_multi_iter(model, next(region_iter), optimizer, accelerator, metric_logger, device)
                image_multi_iter(model, image_batch, optimizer, accelerator, metric_logger, device)
            else:
                if random.random() < 0.5:
                    region_multi_iter(model, next(region_iter), optimizer, accelerator, metric_logger, device)
                else:
                    image_multi_iter(model, image_batch, optimizer, accelerator, metric_logger, device)

        else:
            image_multi_iter(model, image_batch, optimizer, accelerator, metric_logger, device)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_large=optimizer.param_groups[2]["lr"])
        scheduler.step()

        current_epoch = global_step // step_per_epoch

        if (global_step+1) % step_per_epoch == 0:
            if utils.is_main_process():
                train_stats = {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch}

                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if (current_epoch+1) % config['ckpt_frequent'] == 0:
                    model_without_ddp = model
                    if hasattr(model, 'module'):
                        model_without_ddp = model.module

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': scheduler.state_dict(),
                        'config': config,
                        # 'epoch': current_epoch,
                    }
                    checkpointer.save_checkpoint(model_state=save_obj,
                                                 epoch=current_epoch,
                                                 training_states=optimizer.state_dict())

            dist.barrier()

        if (global_step+1) % config['ckpt_frequent_step'] == 0:
            if utils.is_main_process():
                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': scheduler.state_dict(),
                    'config': config,
                    # 'epoch': current_epoch,
                }

                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=current_epoch, step=global_step,
                                             training_states=optimizer.state_dict())

            dist.barrier()

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    config['train_file'] = ','.join(config['train_file'])
    config['train_file_regions'] = ','.join(config['train_file_regions'])

    config['train_file_mono'] = ','.join(config['train_file_mono'])
    config['train_file_text'] = ','.join(config['train_file_text'])  # multilingual parallel texts

    config['batch_size'] = config['images']['batch_size']

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset", flush=True)
    general_dataset, region_dataset, mono_dataset, text_dataset = \
        create_dataset('pretrain_multilingual', config)

    if utils.is_main_process():
        print(f"### train_file: {config['train_file']}", flush=True)
        print(f"### train_file_regions: {config['train_file_regions']}", flush=True)

        print(f"### train_file_mono: {config['train_file_mono']}", flush=True)
        print(f"### train_file_text: {config['train_file_text']}", flush=True)
        print(f"### batch size, {config['batch_size']} x {int(os.environ.get('WORLD_SIZE', 1))}")

    general_loader = torch.utils.data.DataLoader(general_dataset, batch_size=config['images']['batch_size'],
                                               num_workers=config['images']['num_workers'],
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=general_dataset.collate_fn)
    if region_dataset is not None:
        region_loader = torch.utils.data.DataLoader(region_dataset, batch_size=config['regions']['max_images'],  # batch_size = max_images * max_regions
                                                   num_workers=config['regions']['num_workers'],
                                                   pin_memory=True,
                                                   drop_last=False,
                                                   collate_fn=region_dataset.collate_fn)
    else:
        region_loader = None

    if mono_dataset is not None:
        mono_loader = torch.utils.data.DataLoader(mono_dataset, batch_size=config['images_mono']['batch_size'],
                                                  num_workers=config['images_mono']['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  collate_fn=mono_dataset.collate_fn)
    else:
        mono_loader = None

    if text_dataset is not None:
        text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=config['texts_para']['batch_size'],
                                                  num_workers=config['texts_para']['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  collate_fn=text_dataset.collate_fn)
    else:
        text_loader = None

    print("Creating model", flush=True)
    model = CrossViewLM(config=config)
    # print(model)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    arg_sche['step_per_epoch'] = math.ceil(config['train_dataset_size'] / (config['batch_size'] * world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    arg_acc = utils.AttrDict(config['accelerator'])
    accelerator = ApexDDPAccelerator(arg_acc, logger=None)

    model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, world_size, rank)
    reinit_scheduler_properties_mysched(optimizer, lr_scheduler, arg_sche)

    checkpointer = Checkpointer(args.output_dir)

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    epoch_info = (start_epoch, max_epoch)

    print("Start training", flush=True)
    train(model, general_loader, region_loader, mono_loader, text_loader, optimizer, epoch_info, device, lr_scheduler, config,
          accelerator, checkpointer)
    dist.barrier()

    if utils.is_main_process():
        os.system("cat log.txt")
        hcopy('log.txt', args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), flush=True)

    print('### Time {}'.format(total_time_str))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/pretrain')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epoch', default=-1, type=int, help="for pre-training (debug) only")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    hmkdir(args.output_dir)

    yaml.dump(config, open('config.yaml', 'w'))
    hcopy('config.yaml', args.output_dir)

    main(args, config)