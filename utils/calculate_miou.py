# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
from glob import glob

input_dir = '../final_test/road_rgb/results_01'
target_dir = '/home/xaserver1/competition/Tianzhi_final_test/rgb/data0/label'
# target_dir = '/home/xaserver1/competition/Tianzhi2/label'
# target_dir = '/home/xaserver1/competition/Tianzhi_final_test/gray/val/label'
mode = 0 #0表示计算总体，1表示单图计算
classes = 5

def main():
    input_list = glob(os.path.join(input_dir, '*.png'))
    print('num of files: ', len(input_list))
    #计算总的miou
    if mode == 0:
        print('calculate the all miou')
        acc_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()

        for input_file in input_list:
            input_imgName = os.path.split(input_file)[-1]
            input = np.array(Image.open(input_file)).astype('uint8')
            targetName = input_imgName.replace('image','label')
            target_file = os.path.join(target_dir, targetName)
            target = np.array(Image.open(target_file)).astype('uint8')

            acc, pix = accuracy(input, target)
            intersection, union = intersectionAndUnion(input, target, classes)
            acc_meter.update(acc, pix)
            intersection_meter.update(intersection)
            union_meter.update(union)
        # summary
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        for i, _iou in enumerate(iou):
            print('class [{}], IoU: {:.4f}'.format(i, _iou))

        print('[Eval Summary]:')
        print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'.format(np.mean(iou[1:]), acc_meter.average()*100))
    #计算单图的miou
    elif mode == 1:
        print('calculate the single image miou')
        for input_file in input_list:
            print('calculating:', input_file)
            input_imgName = os.path.split(input_file)[-1]
            input = np.array(Image.open(input_file)).astype('uint8')
            target_file = os.path.join(target_dir, input_imgName)
            target = np.array(Image.open(target_file)).astype('uint8')
            acc, pix = accuracy(input, target)
            intersection, union = intersectionAndUnion(input, target, classes)

            iou = intersection / (union + 1e-10)
            for i, _iou in enumerate(iou):
                print('class [{}], IoU: {:.4f}'.format(i, _iou))

            print('[Eval Summary]:')
            print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'.format(np.mean(iou[1:]), acc*100))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

if __name__ == '__main__':
    main()