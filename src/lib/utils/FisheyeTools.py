'''
Referenced:
    RAPiD: https://github.com/duanzhiihao/RAPiD
    cocoapi: https://github.com/cocodataset/cocoapi
'''

import json
import cv2 as cv
from collections import defaultdict
import numpy as np

from pycocotools import cocoeval
from pycocotools import mask as maskUtils


# CEPDOF - WEPDTOF api
class FisheyeEval(cocoeval.COCOeval):
    def __init__(self, gt_json=None, dt_json=None, iouType='bbox'):
        assert iouType in ['bbox', 'segm']
        
        self.gt_json = json.load(open(gt_json, 'r')) if isinstance(gt_json, str) else gt_json
        self.dt_json = json.load(open(dt_json, 'r')) if isinstance(dt_json, str) else dt_json
        self._preprocess_gt_dt()
        self.params = cocoeval.Params(iouType=iouType)
        self.params.imgIds = sorted([img['id'] for img in self.gt_json['images']])
        self.params.catIds = sorted([cat['id'] for cat in self.gt_json['categories']])

        # Initialize some variables which will be modified later
        self.evalImgs = defaultdict(list)   # per-image per-category eval results
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
    
    def _preprocess_gt_dt(self):
        # We are not using 'id' in ground truth annotations because it's useless.
        # However, COCOeval API requires 'id' in both detections and ground truth.
        # So, add id to each dt and gt in the dt_json and gt_json.
        for idx, ann in enumerate(self.gt_json['annotations']):
            ann['id'] = ann.get('id', idx+1)
        
        for idx, ann in enumerate(self.dt_json):
            ann['id'] = ann.get('id', idx+1)

            # Calculate the areas of detections if there is not. category_id
            ann['area'] = ann.get('area', ann['bbox'][2]*ann['bbox'][3])
            ann['category_id'] = ann.get('category_id', 1)

        # A dictionary mapping from image id to image information
        self.imgId_to_info = {img['id']: img for img in self.gt_json['images']}
    

    def _ann_to_rle(self, ann):
        info = self.imgId_to_info[ann['image_id']]
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


    def _prepare(self):
        def _toMask(anns):
            for ann in anns:
                rle = self._ann_to_rle(ann)
                ann['segmentation'] = rle
        
        p = self.params
        gts = [ann for ann in self.gt_json['annotations']]
        dts = self.dt_json
        
        if p.iouType == 'segm':
            _toMask(gts)
            _toMask(dts)

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt.get('ignore', False) or gt.get('iscrowd', False)
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation

        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        

    def computeIoU(self, imgId, catId):
        '''
            Compute iou of all boxes in an image
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]
        
        if p.iouType == 'bbox':
            g = [ann['bbox'] for ann in gt]
            d = [ann['bbox'] for ann in dt]
            info = self.imgId_to_info[imgId]
            w, h = info['width'], info['height']
            iou_func = lambda x, y: iou_rle(x, y, img_size=(h,w))
            
        elif p.iouType == 'segm':
            g = [ann['segmentation'] for ann in gt]
            d = [ann['segmentation'] for ann in dt]
            iscrowd = [int(ann['iscrowd']) for ann in gt]
            iou_func = lambda x, y: maskUtils.iou(x, y, iscrowd)

        ious = iou_func(d, g)
        return ious


def xywha2vertex(box, is_degree, stack=True):
    '''
    Args:
        box: tensor, shape(batch, 5), 5=(cx, cy, w, h, degree)
    Return:
        tensor, shape(batch, 4, 2): topleft, topright, bottomright, bottomleft
    '''
    assert is_degree == False and box.ndim == 2 and box.shape[1] >= 5
    batch = box.shape[0]

    center = box[:,0:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]

    # calculate vertical vector
    verti = np.empty((batch,2))
    verti[:,0] = (h/2) * np.sin(rad)
    verti[:,1] = - (h/2) * np.cos(rad)

    # calculate horizontal vector
    hori = np.empty((batch,2))
    hori[:,0] = (w/2) * np.cos(rad)
    hori[:,1] = (w/2) * np.sin(rad)

    # calculate four vertices
    tl = center + verti - hori # b x 1 x 2
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    return np.concatenate([tl,tr,br,bl], axis=1)


def iou_rle(boxes1, boxes2, img_size=2048, xyxy=False, is_degree=True):
    '''
    Use mask and Run Length Encoding to calculate IOU between rotated bboxes.
    NOTE: rotated bounding boxes format is [cx, cy, w, h, degree].
    Args:
        boxes1: list[list[float]], shape[M,5], 5=(cx, cy, w, h, degree)
        boxes2: list[list[float]], shape[N,5], 5=(cx, cy, w, h, degree)
        img_size: int or list, (height, width)
    Return:
        ious: np.array[M,N], ious of all bounding box pairs
    '''
    if xyxy:
        _boxes1 = np.empty(boxes1.shape)
        _boxes1[:, [0, 1]] = (boxes1[:, [0, 1]] + boxes1[:, [2, 3]])/2
        _boxes1[:, [2, 3]] = boxes1[:, [2, 3]] - boxes1[:, [0, 1]]
        _boxes1[:, -1] = boxes1[:, -1]

        _boxes2 = np.empty(boxes2.shape)
        _boxes2[:, [0, 1]] = (boxes2[:, [0, 1]] + boxes2[:, [2, 3]])/2
        _boxes2[:, [2, 3]] = boxes2[:, [2, 3]] - boxes2[:, [0, 1]]
        _boxes2[:, -1] = boxes2[:, -1]
        
        boxes1 = _boxes1.tolist()
        boxes2 = _boxes2.tolist()
        
    assert isinstance(boxes1, list) and isinstance(boxes2, list)
    # convert bounding boxes to torch.tensor
    boxes1 = np.array(boxes1).reshape(-1, 5)
    boxes2 = np.array(boxes2).reshape(-1, 5)
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))

    # Convert angle from degree to radian
    if is_degree:
        boxes1[:, 4] = boxes1[:, 4] * np.pi / 180
        boxes2[:, 4] = boxes2[:, 4] * np.pi / 180

    b1 = xywha2vertex(boxes1, is_degree=False).tolist()
    b2 = xywha2vertex(boxes2, is_degree=False).tolist()

    h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
    b1 = maskUtils.frPyObjects(b1, h, w)
    b2 = maskUtils.frPyObjects(b2, h, w)
    ious = maskUtils.iou(b1, b2, [0 for _ in b2])

    return ious





