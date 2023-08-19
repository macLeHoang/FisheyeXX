from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import copy

from ..generic_dataset import GenericDataset
from utils.FisheyeTools import FisheyeEval

class FisheyeDataset(GenericDataset):
    max_objs = 128
    default_resolution = [512, 512]
    num_categories = 1
    num_joints = 17
    class_name = ['person']
    cat_ids = {1: 1}

    def __init__(self, opt, split):
        # Load annotations
        # data_dir = os.path.join(opt.data_dir, '')
        img_dir = '/content/DATA'
        ann_path = '/content/gdrive/MyDrive/Fisheye/config.yaml'

        super(FisheyeDataset, self).__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)
        print(f'Loaded {split} {self.num_samples} samples')
    
    def __len__(self):
        return self.num_samples
    
    def run_eval(self, gts, dts, iouType):
        coco_eval = FisheyeEval(gts, dts, iouType)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


        
  
    