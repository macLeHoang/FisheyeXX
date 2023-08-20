from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from ..generic_dataset import GenericDataset
from utils.FisheyeTools import FisheyeEval

class CrowdHumanFisheye(GenericDataset):
    num_categories = 1
    num_joints = 17
    default_resolution = [512, 512]
    max_objs = 128
    class_name = ['person']
    cat_ids = {1: 1}

    def __init__(self, opt, split):
        img_dir = '/content/Images'
        ann_path = '/content/annotations/CrowdHuman_train.json'

        super().__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples
    
    def run_eval(self, gts, dts, iouType):
        coco_eval = FisheyeEval(gts, dts, iouType)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()