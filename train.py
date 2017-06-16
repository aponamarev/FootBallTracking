#!/usr/bin/env python
"""
train.py is a training pipeline for training object detection
Created 6/15/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"


import tensorflow as tf
from tensorflow import placeholder
import numpy as np
from src.COCO_DB import COCO
from src.SmallNet import SmallNet
from src.util import draw_boxes, coco_boxes2xmin_ymin_xmax_ymax, resize_wo_scale_dist

CLASSES = ['person']
ANNOTATIONS_FILE = 'src/coco/annotations/instances_train2014.json'
PATH2IMAGES = 'src/coco/images/train2014'

n_classes=1
batch_sz=8
imshape=(720, 640)



coco = COCO(ANNOTATIONS_FILE, PATH2IMAGES, CLASSES)
net = SmallNet(n_classes, batch_sz, imshape)


gpu_id = 0
train_dir = '' # TODO Define

def main():

    from matplotlib.pyplot import imshow

    for i in range(20):
        im, labels, bboxes = coco.get_sample()

        im, scale = resize_wo_scale_dist(im, imshape)
        bboxes = np.array(bboxes)*scale


        draw_boxes(im, list(map(lambda x: coco_boxes2xmin_ymin_xmax_ymax(im, x),bboxes)), ['person']*len(labels))
        imshow(im)

if __name__=='__main__':
    main()






