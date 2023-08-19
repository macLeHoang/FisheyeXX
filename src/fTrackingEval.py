from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import cv2 as cv
import numpy as np
from progress.bar import Bar
import copy
import argparse
from collections import defaultdict
import glob
import json

from pycocotools import mask as mask_utils
import pycocotools.coco as coco
import motmetrics as mm

import torch

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from dataset.dataset_factory import dataset_factory, get_dataset
from detector import Detector
from utils.image import  affine_transform
from utils.FisheyeTools import xywha2vertex, FisheyeEval

class fPrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.fLoader.load_imgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.get_default_calib = dataset.get_default_calib
    self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        root = img_info['root']
        folder = img_info['folder']
        name = img_info['file_name']
        img_path = os.path.join(img_dir, root, folder, name)
        image = cv2.imread(img_path)
        images, meta = {}, {}

        for scale in opt.test_scales:
            input_meta = {}
            calib = img_info['calib'] if 'calib' in img_info \
                else self.get_default_calib(image.shape[1], image.shape[0])
            input_meta['calib'] = calib
            images[scale], meta[scale] = self.pre_process_func(
                image, scale, input_meta)

        ret = {'images': images, 'image': image, 'meta': meta}
        if 'frame_id' in img_info and img_info['frame_id'] == 1:
            ret['is_first_frame'] = 1
            ret['video_id'] = img_info['video_id']
        return img_id, ret

    def __len__(self):
        return len(self.images)


def prefetch_test(opt):
    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Dataset = dataset_factory[opt.test_dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    
    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)
    
    if opt.load_results != '':
        load_results = json.load(open(opt.load_results, 'r'))
        for img_id in load_results:
            for k in range(len(load_results[img_id])):
                if load_results[img_id][k]['class'] - 1 in opt.ignore_loaded_cats:
                    load_results[img_id][k]['score'] = -1
    else:
        load_results = {}

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process), 
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    results = {}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track']
    avg_time_stats = {t: AverageMeter() for t in time_stats}