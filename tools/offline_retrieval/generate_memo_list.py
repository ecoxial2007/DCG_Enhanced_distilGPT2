import json
import glob
import os

folders = ['./dataset/iu_x-ray/',
           './dataset/mimic_cxr/']# depending on your task

feature_dict = {}

for folder in folders:
    with open(os.path.join(folder, 'annotation.json'), 'r') as f:
        data = json.load(f)
    for value in data['train']:
        report = value['report']
        for image_path in value['image_path']:
            if 'iu_x-ray' in folder:
                feature_path = os.path.join(folder, 'image_features', image_path.replace('.png', '.h5'))
            else:
                _, image_path = os.path.split(image_path)
                feature_path = os.path.join(folder, 'image_features', image_path.replace('.png', '.h5'))

            feature_dict[feature_path] = report
            print(feature_dict[feature_path], report)

with open('dataset/memory_list.json', 'w') as ff:
    json.dump(feature_dict, ff, indent=4)