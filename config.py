# -*- coding: utf-8 -*-

#model paths
lednet_gray_path = './saved_weights/GRAY/gray_model_best.pth'
lednet_path0 = './saved_weights/RGB_5-folder/folder-0.pth'
lednet_path1 = './saved_weights/RGB_5-folder/folder-1.pth'
lednet_path2 = './saved_weights/RGB_5-folder/folder-2.pth'
lednet_path3 = './saved_weights/RGB_5-folder/folder-3.pth'
lednet_path4 = './saved_weights/RGB_5-folder/folder-4.pth'
road_gray_path = './saved_weights/road/gray_256add512_0_5_1115_lr5e-4_bs164_train5_iou.th'
road_rgb_path = './saved_weights/road/rgb_0_5_1114_lr5e-4_bs164_retrain3_iou.th'

lednet_gray = {
    'crop_size':512,
    'stay_size':320,
    'classes':5,
    'use_crf':False,
    'crf_config':{'n_iters':2, 'sxy_gaussian':(1, 1), 'compat_gaussian':8,
                  'sxy_bilateral':(13, 13), 'compat_bilateral':5,
                  'srgb_bilateral':(7, 7, 7)}
}

lednet = {
    'crop_size':512,
    'stay_size':320,
    'classes':5,
    'use_crf':False,
    'crf_config':{'n_iters':1, 'sxy_gaussian':(1, 1), 'compat_gaussian':4,
                  'sxy_bilateral':(55, 55), 'compat_bilateral':5,
                  'srgb_bilateral':(13, 13, 13)}
}

Dlinknet_gray = {
    'crop_size':256,
    'stay_size':128,
    'threshold':0.45, 
}

Dlinknet = {
    'crop_size':512,
    'stay_size':320,
    'threshold':0.5,
}
