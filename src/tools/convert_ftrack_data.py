import json
import os
import glob

def convert_ftrack_data(path):
    json_names = glob.glob(os.path.join(path, '*', 'annotations', '*.json'))
    for json_name in json_names:
        with open(json_name, 'r') as f:
            json_data = json.load(f)
        
        video_name = (json_name.split(os.sep)[-1]).split('.')[0]

        if 'video' not in json_data:
            json_data['video'] = [{
                'id': 1,
                'file_name': video_name,
                'video_len': len(json_data['images'])
            }]

        
        for i in range(len(json_data['images'])):
            json_data['images'][i].update({
                'frame_id': i+1,
                'prev_image_id': json_data['images'][i-1]['id'] if i > 0 else -1,
                'next_image_id': json_data['images'][i+1]['id'] if i < len(json_data['images'])-1 else -1,
                'video_id': video_name
            })
        print('{}: {} images'.format(video_name, len(json_data['images'])))

        with open(json_name, 'w') as f:
            json.dump(json_data, f)

if __name__ == '__main__':
    path = '/content/gdrive/MyDrive/Fisheye/Data'
    convert_ftrack_data(path)

