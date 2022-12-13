#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from configs import get_config_voc, get_config_city, get_config_fish, get_config_water, get_config_disk, get_config_kvasir, get_config_city_4, get_config_voc_person
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from dataloader_city import CityScape
from dataloader_voc import VOC
from dataloader import Dataset_uni
from dataloader import ValPre
from network import Network

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

logger = get_logger()

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
from PIL import Image

default_collate_func = dataloader.default_collate

def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

def get_class_colors(*args):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
    N = 2
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    class_colors = cmap.tolist()
    return class_colors[1:]


class SegEvaluator(Evaluator):
    def __init__(self, dataset, class_num, image_mean, image_std, network,
                 multi_scales, is_flip, devices,
                 verbose=False, save_path=None, show_image=False, config=None):
        self.eval_time = 0
        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.class_num = class_num
        self.image_mean = image_mean
        self.image_std = image_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices
        self.config = config

        self.context = mp.get_context('spawn')
        self.val_func = None
        self.results_queue = self.context.Queue(self.ndata)

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.show_image = show_image

    def func_per_iteration(self, data, device):
        pred = data['data']
        label = data['label']
        name = data['fn']
        pred = self.sliding_eval(pred, self.config.eval_crop_size,
                                 self.config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}

        # if self.save_path is not None:
        #     ensure_dir(self.save_path)
        #     ensure_dir(self.save_path+'_color')

        #     fn = name + '.png' if (not name.endswith(".png") or not name.endswith(".png")) else name

        #     'save colored result'
        #     result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
        #     class_colors = get_class_colors()
        #     palette_list = list(np.array(class_colors).flat)
        #     if len(palette_list) < 768:
        #         palette_list += [0] * (768 - len(palette_list))
        #     result_img.putpalette(palette_list)
        #     result_img.save(os.path.join(self.save_path+'_color', fn))

        # if self.show_image:
        #     colors = self.dataset.get_class_colors
        #     image = img
        #     clean = np.zeros(label.shape)
        #     comp_img = show_img(colors, self.config.background, image, clean,
        #                         label,
        #                         pred)
        #     cv2.imshow('comp_image', comp_img)
        #     cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)
        print(len(dataset.get_class_names(self.config.name)))
        result_line = print_iou(iu, mean_pixel_acc,
                                dataset.get_class_names(self.config.name), True)
        if azure:
            mean_IU = np.nanmean(iu)*100
            run.log(name='Test/Val-mIoU', value=mean_IU)
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--dataset', default='VOC', type=str)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    if args.dataset == 'VOC':
        dataset = VOC
        config = get_config_voc()
    elif args.dataset == 'city':
        dataset = CityScape
        config = get_config_city()
    elif args.dataset == 'voc_person':
        dataset = Dataset_uni
        config = get_config_voc_person()
    elif args.dataset == 'city_4':
        dataset = Dataset_uni
        config = get_config_city_4()
    elif args.dataset == 'fish':
        dataset = Dataset_uni
        config = get_config_fish()
    elif args.dataset == 'water':
        dataset = Dataset_uni
        config = get_config_water()
    elif args.dataset == 'kvasir':
        dataset = Dataset_uni
        config = get_config_kvasir()
    else:
        dataset = Dataset_uni
        config = get_config_disk()

    network = Network(config.num_classes, criterion=None, norm_layer=nn.BatchNorm2d, config=config)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    val_pre = ValPre()
    if dataset == CityScape:
        val_pre = None
    dataset = dataset(data_setting, 'val', val_pre, training=False, dataset_name=args.dataset)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image, config=config)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
