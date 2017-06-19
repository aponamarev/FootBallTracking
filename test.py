#!/usr/bin/env python
"""
test.py is a tool designed to test the trained net.
Created 6/19/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"


from src.SmallNet import SmallNet
import tensorflow as tf
from tensorflow import placeholder
from src.util import check_path


path_to_net = 'logs/net384/model.ckpt'
imshape = (512, 512)

coco_labels=[1, 2, 3, 4]
gpu_id = 0

def process_image(img, net, sess):

    anchor_conf, deltas, classes = sess.run(net)



def main():

    # Create a graph that can place the net to both CPU and GPU
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("gpu:{}".format(gpu_id)):
            net = SmallNet(coco_labels, 1, imshape)

            # Create input placeholders for the net
            im_ph = placeholder(dtype=tf.float32, shape=[*imshape[::-1], 3], name="img")
            labels_ph = placeholder(dtype=tf.float32, shape=[net.WHK, net.n_classes], name="labels")
            mask_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 1], name="mask")
            deltas_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 4], name="deltas_gt")
            bbox_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 4], name="bbox_gt")
            # Grouped those placeholder for ease of handling
            inputs = (im_ph, bbox_ph, deltas_ph, mask_ph, labels_ph)

            net.setup_inputs(*inputs)

            # Initialize variables in the model and merge all summaries
            initializer = tf.global_variables_initializer()
            saver = tf.train.Saver(tf.global_variables())

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True


    sess = tf.Session(config=config, graph=graph)
    sess.run(initializer)

    saver = tf.train.Saver() #Saver(net.weights)
    saver.restore(sess, check_path(path_to_net))

    net.anchor_confidence, net.det_boxes



