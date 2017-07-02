#!/usr/bin/env python
"""
test.py is a tool designed to test the trained net.
Created 6/19/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"


from src.AdvancedNet import AdvancedNet as Net
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow import placeholder
from src.util import check_path, resize_wo_scale_dist, draw_boxes, filter_prediction, bbox_transform
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from matplotlib.pyplot import imshow


path_to_net = 'logs/adv1/model.ckpt'
imshape = (320, 320)
batch_size = 1


img_list = glob('dataset/images/train2014/*.jpg')


coco_labels=[1, 2, 3, 4]
CLASSES = ['person', 'bicycle', 'car', 'motorcycle']
gpu_id = 0


def main():

    # Create a graph that can place the net to both CPU and GPU
    graph = tf.Graph()
    with graph.as_default():
        net = Net(coco_labels, imshape, width=0.5)
        net.optimization_op = 'adam'

        # Create input placeholders for the net
        im_ph = placeholder(dtype=tf.float32, shape=[batch_size,*imshape[::-1], 3], name="img")
        labels_ph = placeholder(dtype=tf.float32, shape=[batch_size,net.WHK, net.n_classes], name="labels")
        mask_ph = placeholder(dtype=tf.float32, shape=[batch_size,net.WHK, 1], name="mask")
        deltas_ph = placeholder(dtype=tf.float32, shape=[batch_size,net.WHK, 4], name="deltas_gt")
        bbox_ph = placeholder(dtype=tf.float32, shape=[batch_size,net.WHK, 4], name="bbox_gt")
        # Grouped those placeholder for ease of handling
        inputs = (im_ph, bbox_ph, deltas_ph, mask_ph, labels_ph)

        net.setup_inputs(*inputs)

        # Initialize variables in the model and merge all summaries
        initializer = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True


    sess = tf.Session(config=config, graph=graph)
    sess.run(initializer)

    restore_variables = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(restore_variables, reshape=True)
    saver.restore(sess, path_to_net)

    for p in img_list:
        im = cvtColor(imread(check_path(p)), COLOR_BGR2RGB)
        im = process(im, net, sess, threshold=0.55, max_obj=50)
        imshow(im.astype(np.uint8))




def process(img, net, sess, threshold=0.5, max_obj=50):



    img, _ = resize_wo_scale_dist(img, imshape)
    p = net.infer(img, sess)

    for i in range(len(p.bboxes)):

        label = []

        final_boxes, final_probs, final_cls_idx = \
            filter_prediction(p.bboxes[i], p.conf[i], p.classes[i],
                              PROB_THRESH=threshold, TOP_N_DETECTIONS=max_obj, n_classes=len(coco_labels))
        for box_id in range(len(final_boxes)):
            kernel_id = final_cls_idx[box_id]
            label.append(CLASSES[kernel_id] + " {}%".format(int(final_probs[box_id] * 100)))

        img = draw_boxes(img,
                         bbox_transform(np.hsplit(np.array(final_boxes),4)),
                         label, thickness=1, fontScale=0.5)

    return img


if __name__=="__main__":
    main()



