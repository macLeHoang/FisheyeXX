# import labelme2coco
import json
from collections import defaultdict


def convert_labelme2coco(json_path, seg_path):
    seg_data = dict()
    seg_names = glob.glob(seg_path)
    for seg_name in seg_names:
        pass

    json_names = glob.glob(os.path.join(path, '*', 'annotations', '*.json'))
    for json_name in json_names:
        with open(json_name, 'r') as f:
            json_data = json.load(f)

        