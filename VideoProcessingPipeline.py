#!/usr/bin/env python
"""
VideoProcessingPipeline is a pipeline desgined to detect football players in the video.
Created 7/1/17 at 11:43 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import tensorflow as tf
import numpy as np
from tensorflow import placeholder
from tensorflow.python.platform.app import flags
from src.AdvancedNet import AdvancedNet as Net
from src.util import resize_wo_scale_dist, filter_prediction, draw_boxes, bbox_transform
from moviepy.editor import VideoFileClip

FLAGS = flags.FLAGS

flags.DEFINE_string("video", "video.mpg", "Provide a relative path to a video file to be processed. Default value: video.mpg")
flags.DEFINE_string("save_to", FLAGS.video, "Provide a relative path for a resulting video. Default value: "+ FLAGS.video)
flags.DEFINE_integer("resolution", 320, "Provide value for rescaling input image. Default value is 320 (320x320).")
flags.DEFINE_string("path_to_net", "logs/adv1/model.ckpt", "Provide a relative path to a folder containing model.ckpt...... Default value: logs/adv1/model.ckpt")


# Define variables for net initialization
CLASSES = ['person', 'bicycle', 'car', 'motorcycle']
coco_labels = [1, 2, 3, 4]
batch_size = 1
imshape = (FLAGS.resolution, FLAGS.resolution)
path_to_net = FLAGS.path_to_net


compute = {'sess': None, 'net': None}

def define_net():
    """
    function create a computational graph and session
    :return: sess, net
    """

    # Create a graph that can place the net to both CPU and GPU
    graph = tf.Graph()
    with graph.as_default():
        net = Net(coco_labels, imshape, width=0.5)
        net.optimization_op = 'adam'

        # Create input placeholders for the net
        im_ph = placeholder(dtype=tf.float32, shape=[batch_size, *imshape[::-1], 3], name="img")
        labels_ph = placeholder(dtype=tf.float32, shape=[batch_size, net.WHK, net.n_classes], name="labels")
        mask_ph = placeholder(dtype=tf.float32, shape=[batch_size, net.WHK, 1], name="mask")
        deltas_ph = placeholder(dtype=tf.float32, shape=[batch_size, net.WHK, 4], name="deltas_gt")
        bbox_ph = placeholder(dtype=tf.float32, shape=[batch_size, net.WHK, 4], name="bbox_gt")
        # Grouped those placeholder for ease of handling
        inputs = (im_ph, bbox_ph, deltas_ph, mask_ph, labels_ph)

        print("Setting up a net. It may take a couple of minutes.")
        net.setup_inputs(*inputs)

        # Initialize variables in the model and merge all summaries
        initializer = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
    print("Creating a session.")
    sess = tf.Session(config=config, graph=graph)
    sess.run(initializer)
    print("Restoring a model.")
    restore_variables = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(restore_variables, reshape=True)
    saver.restore(sess, path_to_net)
    print("Pipeline is ready for video processing.")

    return sess, net


def process(img, threshold=0.6, max_obj=20):


    img, _ = resize_wo_scale_dist(img, imshape)
    p = compute['net'].infer(img, compute['sess'])

    for i in range(len(p.bboxes)):

        label = []

        final_boxes, final_probs, final_cls_idx, anch_ids = \
            filter_prediction(p.bboxes[i], p.conf[i], p.classes[i], PROB_THRESH=threshold, TOP_N_DETECTIONS=max_obj)
        for box_id in range(len(final_boxes)):
            kernel_id = final_cls_idx[box_id]
            label.append(CLASSES[kernel_id] + " {}%".format(int(final_probs[box_id] * 100)))

        img = draw_boxes(img, bbox_transform(np.hsplit(final_boxes,4)), label, thickness=1, fontScale=0.5)

    return img

def main():

    compute['sess'], compute['net'] = define_net()

    clip = VideoFileClip(FLAGS.video)
    processed_clip = clip.fl_image(process)
    processed_clip.write_videofile(FLAGS.save_to, audio=False)


if __name__ == "__main__":
    main()










