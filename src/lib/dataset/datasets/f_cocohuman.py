from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
import numpy as np
import json
import os
import copy
from PIL import Image
from collections import defaultdict

from utils.image import affine_transform
from ..generic_dataset import GenericDataset
from utils.FisheyeTools import FisheyeEval

class COCOHumanFisheye(GenericDataset):
    max_objs=128
    default_resolution=[512, 512]
    num_categories = 1
    num_joints = 17
    class_name=['person']
    cat_ids={1: 1}

    def __init__(self, opt, split):
        # Load annotations
        # data_dir = os.path.join(opt.data_dir, 'cocohumanfisheye')
        # img_dir = os.path.join(data_dir, 'images')
        # ann_path = 'path to .json'

        ann_path = '/content/annotations/instances_train2017.json'
        img_dir = '/content/train2017'

        super(COCOHumanFisheye, self).__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)
        print(f'Loaded {split} {self.num_samples} samples')
    
    def __len__(self):
        return self.num_samples

    def _to_float(self, x):
        return float(f"{x:.2f}")
    
    def run_eval(self, gts, dts, iouType, eval_type='Detect'):
        assert eval_type in ['Detect', 'Track']
        if eval_type == 'Detect':
            coco_eval = FisheyeEval(gts, dts, iouType)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        