# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image
from glob import glob

label_dir = './results'
output_dir = './color_results'
COLOR_MAP = {'0': [0, 0, 0], '1': [255, 0, 0], '2': [0, 255, 0], '3': [0, 0, 255],'4': [255, 255, 0]}

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

imgPaths = glob(os.path.join(label_dir, '*.png'))
print('Num of files: ', len(imgPaths))

for imgPath in imgPaths:
    label = np.array(Image.open(imgPath))
    imageName = os.path.split(imgPath)[-1]
    label_color = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(5):
        label_color[label == i] = COLOR_MAP[str(i)]
    #label_color = get_color(label)
    label_color = Image.fromarray(label_color.astype('uint8'))
    label_color.save(os.path.join(output_dir, 'color_'+imageName))

'''
def get_color(result):
    pre_label = np.zeros(shape=(result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for i in range(5):
        pre_label[result == i] = COLOR_MAP[str(i)]
    return pre_label
'''