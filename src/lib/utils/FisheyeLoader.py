'''
    This module has just been built for convenience when loading datas for my thesis
'''

import glob
from collections import defaultdict
import os
import json
import yaml
import time

from pycocotools import mask as maskUtils

class FisheyeLoader:
    def __init__(self, annotation_dir=None, sample_rate=-1, split=None):
        '''
            :param annotation_dir (str): location of annotation file
            :param sample_rate (int): because imgs are taken from high FPS videos so
                                      it do not need to use all the imgs
        '''
        if not annotation_dir is None:
            print('loading annotations into memory...')
            tic = time.time()

            if True:
                print('This loader is used for my project only !!! Refer to FisheyeLoader.py to fit yours :>')
                annotation_files = self.load_config_YAML(annotation_dir, split)
            
            # annotation_files = glob.glob(os.path.join(annotation_dir, '*.json'))

            self.dataset, self.imgs, self.imgIds, self.anns = self.load_data(annotation_files)
            if sample_rate > 0:
                self.imgIds = [self.imgIds[i] for i in range(0, len(self.imgIds), sample_rate)]
            assert type(self.dataset)==dict, 'annotation file format {} not supported'.format(type(self.dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
    
    def load_data(self, annotation_files):
        print('creating index...')
        dataset = defaultdict(list)
        imgs = dict()
        imgIds = list()
        annotations = defaultdict(list)

        for ann_files in annotation_files:
            root, _, folder = ann_files.split(os.sep)[-3:]
            folder = folder.split('.')[0]
            with open(ann_files, 'r') as f:
                anns = json.load(f)
            if 'annotations' in anns:
                for ann in anns['annotations']:
                    ann['track_id'] = ann['person_id'] 
                    dataset[ann['image_id']].append(ann)
                annotations['annotations'] += anns['annotations']
            
            if 'images' in anns:
                for i, img in enumerate(anns['images']):
                    img.update({
                        # used to load images
                        'root': root,
                        'folder': folder,

                        # used for tracking task
                        'frame_id': i+1,
                        'prev_image_id': anns['images'][i-1]['id'] if i > 0 else '-1', 
                        'next_image_id': anns['images'][i+1]['id'] if i < len(anns['images'])-1 else '-1',
                        'video_id': folder
                    })
                    imgs[img['id']] = img   
                annotations['images'] += anns['images']  
        imgIds = list(dataset.keys())  
        print('index created!')
        return dict(dataset), imgs, list(sorted(imgIds)), annotations

    def _ann_to_rle(self, ann):
        info = self.imgs[ann['image_id']]
        w, h = info['width'], info['height']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def ann_to_mask(self, ann):
        maskRLE = self._ann_to_rle(ann)
        return maskUtils.decode(maskRLE)
    
    def get_imgIds(self):
        return self.imgIds
    
    def load_imgs(self, imgIds):
        return [self.imgs[_] for _ in imgIds]
    
    def load_anns(self, imgIds):
        anns = []
        for imgId in imgIds:
            anns += self.dataset.get(imgId, [])
        return anns
    
    def load_config_YAML(self, config_path='', split='train'):
        with open(config_path, 'r') as f:
            try:
                data_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        if 'path' in data_config:
            data_path = data_config['path']
        
        total_json_files = []

        if split == 'train':
            if 'train' in data_config:
                for folder in data_config['train']:
                    json_files = [os.path.join(data_path, folder,'annotations', f'{f}.json') 
                                        for f in data_config['train'][folder]]
                    total_json_files += json_files

        if split == 'test':
            if 'test' in data_config:
                for folder in data_config['test']:
                    json_files = [os.path.join(data_path, folder,'annotations', f'{f}.json') 
                                        for f in data_config['test'][folder]]
                    total_json_files += json_files

        return total_json_files
    
    
    def showResults(self, results):
        pass