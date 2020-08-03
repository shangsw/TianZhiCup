import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time
# from networks.unet import Unet
# from networks.dunet import Dunet
from models.dlinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool

from PIL import Image
from skimage.io import imsave

from albumentations import RandomContrast, RandomBrightness
from albumentations import HorizontalFlip, VerticalFlip, Transpose, RandomRotate90,ShiftScaleRotate,RandomSizedCrop,LongestMaxSize
import config

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
    
    def test_one_img_from_path(self, img):               
        
        self.net.eval()
       
        # img = cv2.imread(path)#.transpose(2,0,1)[None]
        # img = np.array(img)[:,:,::-1]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3, img4])
        #random contrast
        img_contrast = []
        # contrast_set = [(0.01,0.02),(0.03,0.04),(0.05,0.06),(0.07,0.08)]
        contrast_set = []
        if len(contrast_set) > 0:
            for i in contrast_set:
                img_contrast.append(RandomContrast(limit=i, p=1)(image=img)['image'])
            # for i in range(4):
            #     img_contrast.append(RandomContrast(limit=(0,1.0), p=1)(image=img)['image'])
            img5 = np.concatenate([img5, np.array(img_contrast)])

        input_img = np.array(img5, np.float32).transpose(0,3,1,2)/255.0 * 3.2 -1.6
        input_img = V(torch.Tensor(input_img).cuda())
        with torch.no_grad():
            mask = self.net.forward(input_img).squeeze().cpu().data.numpy()  #.squeeze(1)
        aug_times = mask.shape[0]
        mask_1, mask_2 = mask[:8], mask[8:]

        mask1 = mask_1[:4] + mask_1[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        mask4 = np.sum(mask_2, axis=0)
        mask_final = mask3 + mask4
        # img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
        # img = V(torch.Tensor(np.array(img, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        # mask3 = self.net.forward(img).squeeze().cpu().data.numpy()
        return (mask_final / aug_times)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

def get_pred_label(image, solver, crop_size, stay_size, mode):
    '''
    params: 要求crop_size与stay_size为同奇偶
    img_path: 测试图片的路径
    crop_size: 图片太大需要裁切的大小；为一个整数c，代表(c,c)区域。
    stay_size: 最终保存结果的size，是crop_size的中间部分；为一个整数s，代表(s,s)区域
    '''
    pad_sts_row = int(stay_size - image.shape[0] % stay_size)
    pad_sts_column = int(stay_size - image.shape[1] % stay_size)
    
    pad_cs_row = (crop_size - stay_size) // 2
    pad_cs_column = (crop_size - stay_size) // 2
    
    pad = ((pad_cs_row, pad_sts_row + pad_cs_row), (pad_cs_column, pad_sts_column + pad_cs_column), (0,0))
    pad_img = np.pad(image, pad, mode='reflect')
    pred_label = np.zeros(shape=(image.shape[0]+pad_sts_row, image.shape[1]+pad_sts_column))
    # predict and get the label
    for row_idx in range(0, pad_img.shape[0], stay_size):
        if row_idx > pad_img.shape[0] - crop_size:
            break
        for column_idx in range(0, pad_img.shape[1], stay_size):
            if column_idx > pad_img.shape[1] - crop_size:
                break
            crop_img = pad_img[row_idx:row_idx + crop_size, column_idx:column_idx + crop_size,:]
            if mode == 'gray':
                resize_img = LongestMaxSize(512)(image=crop_img)['image']
                crop_result = solver.test_one_img_from_path(resize_img)
                crop_mask = LongestMaxSize(256)(image=crop_result)['image'] #这里是mask的resize
                # crop_mask[crop_mask > 0.45] = 1.0
                # crop_mask[crop_mask <= 0.45] = 0.0
            elif mode == 'rgb':
                crop_mask = solver.test_one_img_from_path(crop_img)
                # crop_mask[crop_mask > 0.5] = 1.0
                # crop_mask[crop_mask <= 0.5] = 0.0
            
            pred_label[row_idx:row_idx+stay_size, 
                       column_idx:column_idx+stay_size] = crop_mask[pad_cs_row:pad_cs_row+stay_size, 
                                                                      pad_cs_column:pad_cs_column+stay_size]
    final_pred = pred_label[:image.shape[0], :image.shape[1]]
    return final_pred

def road_extract(image, model_path, mode):
    solver = TTAFrame(DinkNet34)
    solver.load(model_path)
    if mode == 'gray':
        image = np.array([image, image, image]).transpose(1,2,0)
        pred_result = get_pred_label(image, solver, config.Dlinknet_gray['crop_size'], config.Dlinknet_gray['stay_size'], 'gray')
    elif mode == 'rgb':
        image = image[:,:,::-1] # RGB to BGR
        pred_result = get_pred_label(image, solver, config.Dlinknet['crop_size'], config.Dlinknet['stay_size'], 'rgb')
    return pred_result


if __name__ == '__main__':

    solver = TTAFrame(DinkNet34)
    solver.load('./weights/gray_256add512_0_30_0.th')

    source = './dataset/gray_256resize512_1/val/src_512/'
    for i in os.listdir(source):
        # src = Image.open(source + i)
        # src = np.array(src)
        src = cv2.imread(source + i)
        
        predict_result = get_pred_label(src, 256, 256)
        predict_result_path = './predict_result/gray_256add512_0_30_0/'
        if not (os.path.exists(predict_result_path)):
            os.mkdir(predict_result_path)
        imsave(predict_result_path+i.replace('image','label').replace('tif','png'), predict_result.astype(np.uint8))




