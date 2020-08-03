# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
from glob import glob
from PIL import Image
import time
from test_lednet import test_lednet
from utils.dense_crf import dense_crf
from road_extract import road_extract
import config


COLOR_MAP = {'0': [0, 0, 0], '1': [255, 0, 0], '2': [0, 255, 0], '3': [0, 0, 255],'4': [255, 255, 0]}

parser = argparse.ArgumentParser(description='fusion the models to get final results')
# Datasets
parser.add_argument('-d', '--test_dir', default='/home/xaserver1/competition/Tianzhi2/GRAY_ori/src', type=str, help='path to dataset')
parser.add_argument('-o', '--output_dir', default='.', type=str, help='path to saving results')
parser.add_argument('-m', '--mode', default=0, type=int, help='fusion strategy')    #0: belifs add; 1: vote
parser.add_argument('--visualize', dest='visualize', action='store_true', help='whether visualize results')
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

outraw_dir = os.path.join(args.output_dir, 'results_01')
outvis_dir = os.path.join(args.output_dir, 'results_vis')
outrgb_dir = os.path.join(args.output_dir, 'results_rgb')
if not os.path.isdir(outraw_dir):
    os.makedirs(outraw_dir)
if args.visualize:
    if not os.path.isdir(outvis_dir):
        os.makedirs(outvis_dir)
if not os.path.isdir(outrgb_dir):
    os.makedirs(outrgb_dir)

def main():
    start = time.time()
    imgPaths = glob(os.path.join(args.test_dir, '*.tif'))
    print('Num of files: ', len(imgPaths))
    for imgPath in imgPaths:
        image = np.array(Image.open(imgPath))
        print('-'*10,'predicting image: ', imgPath, '-'*10)
        imageName = os.path.split(imgPath)[-1]
        outName = imageName.replace('.tif', '.png')
        if len(imageName) > 15:
            continue
        if len(image.shape) == 2:
            #gray image predict
            result = test_lednet(image, config.lednet_gray_path, 'gray')
            result = crf(result, image, config.lednet_gray['use_crf'], **config.lednet_gray['crf_config'])
            # result = morph_close(result)   #闭操作
            result = np.argmax(result, axis=2).astype('uint8')
            result[result==2] = 0   #set road to 0
            print('gray time:', time.time()-start)
            start = time.time()
            #extract road
            road_result = road_extract(image, config.road_gray_path, 'gray')
            result[road_result>config.Dlinknet_gray['threshold']] = 2
            print('gray road time:', time.time()-start)
            start = time.time()
        else:
            #rgb image predict
            results_list = []
            #predicting lednet_folder0
            lednet0 = test_lednet(image, config.lednet_path0, 'rgb')
            results_list.append(lednet0)
            #predicting lednet_folder1
            lednet1 = test_lednet(image, config.lednet_path1, 'rgb')
            results_list.append(lednet1)
            #predicting lednet_folder2
            lednet2 = test_lednet(image, config.lednet_path2, 'rgb')
            results_list.append(lednet2)
            #predicting lednet_folder3
            lednet3 = test_lednet(image, config.lednet_path3, 'rgb')
            results_list.append(lednet3)
            #predicting lednet_folder4
            lednet4 = test_lednet(image, config.lednet_path4, 'rgb')
            results_list.append(lednet4)
            
            if args.mode == 0:
                result = np.array(results_list).mean(axis=0)
                #if use CRF
                result = crf(result, image, config.lednet['use_crf'], **config.lednet['crf_config']) 
                # result = morph_close(result)   #闭操作
                result = np.argmax(result, axis=2).astype('uint8')
                print('mean time:', time.time()-start)
                start = time.time()
            elif args.mode == 1:    #vote
                # results_list = list(map(morph_close, results_list))   #闭操作
                result = vote(results_list)
                print('vote time:', time.time()-start)
                start = time.time()
            #result:(512,512), int:0~4
            result[result==2] = 0   #set road to 0
            #road extract
            road_result = road_extract(image, config.road_rgb_path, 'rgb')
            # road_result = crf(road_result, image, config.Dlinknet['use_crf'], **config.Dlinknet['crf_config'])
            result[road_result>config.Dlinknet['threshold']] = 2
            print('road time:', time.time()-start)
            start = time.time()
        
        color_result = get_color(result)
        
        Image.fromarray(result.astype('uint8')).save(os.path.join(outraw_dir, outName))
        Image.fromarray(color_result.astype('uint8')).save(os.path.join(outrgb_dir, outName))

        if args.visualize:
            if len(image.shape) == 2:
                image = np.array([image, image, image]).transpose(1,2,0)
            vis_result = image.copy()
            vis_result[result!=0] = color_result[result!=0]*0.3 + image[result!=0]*0.7
            Image.fromarray(vis_result.astype('uint8')).save(os.path.join(outvis_dir, outName))
        # print('Times: ', time.time()-start)


def crf(probs, image, flag=False, **crf_config):
    img = image.copy()
    if flag:
        if len(img.shape) == 2:
            img = np.array([img, img, img]).transpose(1,2,0).copy(order='C')
        if len(probs.shape) == 2:
            probs = np.tile(probs[:,:,np.newaxis], (1,1,2))
            probs[:,:,0] = 1 - probs[:,:,1] 
        crf_result = dense_crf(probs=probs.astype('float32'), img=img, **crf_config)
        if crf_result.shape[2] == 2:
            crf_result = crf_result[:,:,1]
    else:
        crf_result = probs

    return crf_result


def vote(result_list):
    h, w, classes = result_list[0].shape
    result_list = np.reshape(np.array(result_list), (-1, h*w, classes))
    result = np.argmax(np.array(result_list), axis=-1)
    result = np.reshape(result, (result.shape[0], h*w)).T
    result_final = np.zeros(result.shape[0])
    for i, label in enumerate(result):
        count = np.bincount(label)
        if np.max(count) > 2:
            result_final[i] = np.argmax(count)
        else:
            result_final[i] = np.argmax(count[1:])
    result_final = np.reshape(result_final.T, (h, w))
    return result_final
    
def morph_close(pred):
    for i in range(pred.shape[2]):
        pred[:,:,i] = cv2.morphologyEx(pred[:,:,i], cv2.MORPH_CLOSE, np.ones((17,17), np.uint8)) 
    return pred   

def get_color(result):
    pre_label = np.zeros(shape=(result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for i in range(5):
        pre_label[result == i] = COLOR_MAP[str(i)]
    return pre_label


if __name__ == '__main__':
    main()
    print('Complete!')
