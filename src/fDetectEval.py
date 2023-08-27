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

from pycocotools import mask as mask_utils
import pycocotools.coco as coco
import json

import torch

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from dataset.dataset_factory import dataset_factory, get_dataset
from detector import Detector
from utils.image import  affine_transform
from utils.FisheyeTools import xywha2vertex, FisheyeEval

class fDetectDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset):
        self.images = dataset.images
        if opt.fisheye and opt.fisheye_pretrain:
            self.load_image_func = dataset.coco.loadImgs
            self.anns_loader = dataset.coco
        else:
            self.load_image_func = dataset.fLoader.load_imgs
            self.anns_loader = dataset.fLoader # TODO

        self.img_dir = dataset.img_dir
        # self.pre_process_func = pre_process_func
        # self.get_default_calib = dataset.get_default_calib
        self.opt = opt
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.opt.fisheye and self.opt.fisheye_pretrain:
            img_id = self.images[index]
            img_info = self.load_image_func(ids=[img_id])[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
        else:
            img_id = self.images[index]
            img_info = self.load_image_func(imgIds=[img_id])[0]
            root = img_info['root']
            folder = img_info['folder']
            name = img_info['file_name']
            img_path = os.path.join(self.img_dir, root, folder, name)
        
        img = cv.imread(img_path)

        if self.opt.fisheye_pretrain:
            ann_ids = self.anns_loader.getAnnIds(imgIds=[img_id], catIds=[1])
            anns = copy.deepcopy(self.anns_loader.loadAnns(ids=ann_ids))
        else:
            anns = self.anns_loader.load_anns(imgIds=[img_id]) # TODO CHECK 


        expand = self.opt.fisheye_pretrain
        test_rand_scale = self.opt.rand_scale_test # not support yet
        rand_rot_test = self.opt.rand_rot_test
        img, anns = self._load_test_data(img, anns, expand, rand_rot_test, test_rand_scale)

        return {'img':img, 'img_id':img_id, 
                'img_path': img_path, 'anns':anns}
        

    def _load_test_data(self, img, anns, expand=False, 
                           rand_rot_test=False,
                           test_rand_scale=False):
        '''Load img and anns for testing 
           Arguments: 
                - random rotating: Optional
                - multi scale testing: Not support yet
        '''
        if rand_rot_test:
            h, w = img.shape[:2]
            rot = np.random.randint(-180, 180)
            if expand:
                new_w, new_h = self._new_rotated_wh(rot, w, h)
                translate_mat = np.array([[1., 0., (new_w-w)//2], [0., 1., (new_h-h)//2]])
            else:
                new_w, new_h = w, h
                translate_mat = np.array([[1., 0., 0.], [0., 1., 0.]])

            scale = np.random.choice(np.arange(0.6, 1.4, 0.1)) if self.opt.rand_scale_test else 1
            rot_mat = cv.getRotationMatrix2D((new_w/2, new_h/2), rot, scale)
            img = cv.warpAffine(img, translate_mat, (int(new_w), int(new_h)))
            img = cv.warpAffine(img, rot_mat, (int(new_w), int(new_h)))
        else:
            rot=0

        # _new_w = (int(new_w) // 32 + 1) * 32
        # _new_h = (int(new_h) // 32 + 1) * 32

        # padding_im = np.zeros((_new_h, _new_w, 3))
        # padding_im[:int(new_h), :int(new_w)] = img
        # img = padding_im
        # new_w, new_h = _new_w, _new_h
        
        new_anns = []

        for ann in anns:
            category_id = ann['category_id']
            if self.opt.fisheye_pretrain:
                x, y, w, h = ann['bbox'] # x topleft, y topleft, w, h
                cx, cy = x + w/2, y + h/2
                a = -rot
            else:
                cx, cy, w, h, a = ann['bbox'] # x center, y center, w, h
                a = a - rot

            if rand_rot_test:
                cxcy = np.array([cx, cy], dtype=np.float32)
                cxcy =  affine_transform(cxcy, translate_mat)
                cxcy =  affine_transform(cxcy, rot_mat)
            else:
                cxcy = [cx, cy]

            if self.opt.seg:
                if self.opt.fisheye_pretrain:
                    instance_mask = self.anns_loader.annToMask(ann)
                else:
                    instance_mask = self.anns_loader.ann_to_mask(ann)

                if rand_rot_test:
                    instance_mask = cv.warpAffine(instance_mask, translate_mat,
                                                (int(new_w), int(new_h)),
                                                flags=cv.INTER_LINEAR)
                    instance_mask = cv.warpAffine(instance_mask, rot_mat, 
                                                (int(new_w), int(new_h)), 
                                                flags=cv.INTER_LINEAR)
                instance_mask = np.asfortranarray(instance_mask).astype(np.uint8)
                instance_mask = mask_utils.encode(instance_mask)
                instance_mask['counts'] = instance_mask['counts'].decode("utf-8")

            new_anns.append({
                'category_id': category_id, 
                'bbox': [cxcy[0], cxcy[1], w, h, a],
                'iscrowd': ann['iscrowd'],
                'segmentation': instance_mask if self.opt.seg else []
            })
        
        return img, new_anns

    def _new_rotated_wh(self, rot, w, h):
        assert -180 <= rot < 180
        if 0 <= rot < 90:
            rot_radian = rot / 180 * np.pi
            new_w = w*np.cos(rot_radian) + h*np.sin(rot_radian)
            new_h = w*np.sin(rot_radian) + h*np.cos(rot_radian)
        elif 90 <= rot < 180:
            rot_radian = (rot-90) / 180 * np.pi
            new_w = h*np.cos(rot_radian) + w*np.sin(rot_radian)
            new_h = h*np.sin(rot_radian) + w*np.cos(rot_radian)
        elif -90 <= rot < 0:
            rot_radian = (rot+90) / 180 * np.pi
            new_w = h*np.cos(rot_radian) + w*np.sin(rot_radian)
            new_h = h*np.sin(rot_radian) + w*np.cos(rot_radian)
        elif -180 <= rot < -90 :
            rot_radian = (rot+180) / 180 * np.pi
            new_w = w*np.cos(rot_radian) + h*np.sin(rot_radian)
            new_h = w*np.sin(rot_radian) + h*np.cos(rot_radian)
        return new_w, new_h

def fDtect_test(opt):
    gts = defaultdict(list)
    dts = []

    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    
    detector = Detector(opt)
    Dataset = dataset_factory[opt.test_dataset]
    dataset = Dataset(opt, 'test')
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    data_loader = torch.utils.data.DataLoader(
                        fDetectDataset(opt, dataset), 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=1, 
                        pin_memory=True
    )
    num_iters = len(data_loader)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)

    for ind, meta in enumerate(data_loader):
        img = meta['img'] 
        img_id = meta['img_id'] 
        img_path = meta['img_path']
        gt_anns = meta['anns']

        img = img[0].numpy()
        ret = detector.run(img)
        results = ret['results']

        for ann in gt_anns:
            ann['bbox'] = list(map(float, ann['bbox']))
            if opt.seg:
                ann['segmentation']['size'] = list(map(float, ann['segmentation']['size']))
                ann['segmentation']['counts'] = ann['segmentation']['counts'][0]
            gts['annotations'].append({
                'area': ann['bbox'][2]*ann['bbox'][3],
                'bbox': ann['bbox'],
                'category_id': 1,
                'image_id': int(img_id[0]) if opt.fisheye_pretrain else img_id[0],
                'iscrowd': int(ann['iscrowd'][0]),
                'segmentation': ann['segmentation'] if opt.seg else [],
                'person_id': -1
            })
        gts['images'].append({
            'file_name': os.path.split(img_path[0])[-1],
            'id': int(img_id[0]) if opt.fisheye_pretrain else img_id[0], 
            'width': int(img.shape[1]),
            'height': int(img.shape[0])
        })
    
        for result in results:
            x1, y1, x2, y2, a = result['bbox']
            bbox = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, a / np.pi * 180] # angle is degree
            bbox = list(map(lambda x: float(f"{x:.2f}"), bbox))
            detection = {
                "image_id": int(img_id[0]) if opt.fisheye_pretrain else img_id[0],
                "category_id": 1,
                "bbox": bbox,
                "score": float("{:.2f}".format(result['score'])),
                "segmentation": result['pred_mask'] if opt.seg else []
            }
            dts.append(detection)

        
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                        ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        if opt.print_iter > 0:
            if ind % opt.print_iter == 0:
                print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
        else:
            bar.next()

    gts['categories'] = [{
            "id": 1,
            "name": "person",
            "supercategory": "person"
        }]
    bar.finish()

    # eval_type='Detect' if not opt.tracking else 'Track'
    dataset.run_eval(gts, dts, 'bbox')
    if opt.seg:
        dataset.run_eval(gts, dts, 'segm')

if __name__ == '__main__':
    opt = opts().init()
    fDtect_test(opt)
    
