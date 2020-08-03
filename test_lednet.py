import cv2
import numpy as np
import torch
import os
import importlib

from PIL import Image
from argparse import ArgumentParser
from skimage.io import imsave
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage


import argparse
from albumentations import RandomBrightness,RandomBrightnessContrast,RandomContrast

from models.lednet import Net as Net_rgb
from models.lednet_gray import Net as Net_gray
from albumentations import VerticalFlip,HorizontalFlip,Transpose
import config

# NUM_CLASSES = 5

class Test:
    def __init__(self, model_path, size, channels, classes, stay_size, model,aug):
        self.channels = channels
        self.classes = classes
        self.size = size
        self.aug = aug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        weightspath = model_path
        
        # model = Net(self.classes, self.channels)

        model = torch.nn.DataParallel(model)
        # if (not args.cpu):
        model = model.cuda()
        self.model = self.load_my_state_dict(model, torch.load(weightspath))
        # print("Model and weights LOADED successfully")

        self.model.eval()

        self.stay_size = stay_size
        

    def normal(self, image):
        return (image-np.average(image)) / np.std(image)

    
    def load_my_state_dict(self,model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        return model

    def predict(self, img):
  
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        pad0 = int(np.ceil(img.shape[0] / self.stay_size) * self.stay_size - img.shape[0])
        pad1 = int(np.ceil(img.shape[1] / self.stay_size) * self.stay_size - img.shape[1])

        pad00 = int((self.size - self.stay_size) / 2)
        pad11 = int((self.size - self.stay_size) / 2)

        pad = ((pad00, pad0+pad00), (pad11, pad1+pad11), (0, 0))

        pad_image = np.pad(img, pad, mode='reflect')
        pre_label = np.zeros(shape=(img.shape[0]+pad0, img.shape[1]+pad1, self.classes))

        for dim0 in range(0, pad_image.shape[0], self.stay_size):
            if dim0 > pad_image.shape[0] - self.size:
                    break
            for dim1 in range(0, pad_image.shape[1], self.stay_size):
                if dim1 > pad_image.shape[1] - self.size:
                    break
                crop_image_ori = pad_image[dim0:dim0 + self.size, dim1:dim1 + self.size, :]

                #图像归一化
                crop_image = crop_image_ori / 255.0
                #gray image
                if self.channels == 1:
                    if self.aug:
                        crop_image_list = []
                        for i in range(5):
                            crop_image_list.append(RandomContrast(p=1, limit=(0,0.15))(image=crop_image)['image'])
                        crop_image = np.array(crop_image_list)
                    else:
                        crop_image = np.expand_dims(crop_image, axis=0)
                #rgb image
                elif self.channels == 3:
                    # if self.aug:
                    #     crop_image_list = []
                    #     for i in range(5):
                    #         crop_image_list.append(RandomContrast(p=1, limit=0.15)(image=crop_image)['image'])
                    #     crop_image = np.array(crop_image_list)
                    # else:
                    #     crop_image = np.expand_dims(crop_image, axis=0)
                    crop_image = np.expand_dims(crop_image, axis=0)
                
                aug_crop_image = torch.from_numpy(crop_image.transpose((0,3,1,2))).float().to(device=self.device)
                inputs = Variable(aug_crop_image)
                
                with torch.no_grad():
                    outputs = self.model(inputs)
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                result = np.mean(outputs.detach().cpu().numpy(),axis=0).transpose(1,2,0)
            
                # print(result.shape)
                pre_label[dim0:dim0 + self.stay_size, dim1:dim1 + self.stay_size,:] += result[
                                                                                         pad00:pad00 + self.stay_size,
                                                                                         pad11:pad11 + self.stay_size,:]

        pre_label = pre_label[:img.shape[0], :img.shape[1], :]
        return pre_label

def test_lednet(image, model_path, mode='rgb'):
    if mode == 'rgb':
        model = Net_rgb(config.lednet['classes'], 3)
        test = Test(model_path, config.lednet['crop_size'], 3,
                config.lednet['classes'], config.lednet['stay_size'], model, aug=False)
    elif mode == 'gray':
        model = Net_gray(config.lednet_gray['classes'])
        test = Test(model_path, config.lednet_gray['crop_size'], 1,
                config.lednet_gray['classes'], config.lednet_gray['stay_size'], model, aug=True)
    else:
        print('mode must be \'gray\' or \'rgb\'')

    # print('Inferring data using %s' % model_path)
    pre_logits = test.predict(image)
    return pre_logits


'''
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='/home/gpuserver/competition/Tianzhi/src1/', help='test image path')
    parser.add_argument('--label_path', default='/home/gpuserver/competition/Tianzhi/label1/', help='test label path')
    parser.add_argument('--model_path', default='../save_gray_Focal_LS_G/logs/model_best.pth')
    parser.add_argument('--save_path', default='./save_color/', help='save predict result')
    parser.add_argument('--size', default=512, type=int, help='predict size')
    parser.add_argument('--channels', default=1, type=int)
    parser.add_argument('--classes', default=5, type=int)
    parser.add_argument('--stay_size', default=80,type=int, help='stay center size from crop')
    parser.add_argument('--model',default='lednet',help='the name of model')
    parser.add_argument('--aug',default=False,help='if aug data')
    args = parser.parse_args()
    print(args.channels,args.size,args.stay_size,'aug is',args.aug,'label smoothing & Gaussian filter & RC & max')
    test = Test(args.image_path, args.label_path, args.model_path, args.save_path, args.size, args.channels,
                args.classes, args.stay_size,args.model,args.aug)
    test.predict()
'''
