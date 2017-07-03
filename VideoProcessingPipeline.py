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
try:
    from moviepy.editor import VideoFileClip
except:
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip

flags.DEFINE_string("video", "video.mp4", "Provide a relative path to a video file to be processed. Default value: video.mpg")
flags.DEFINE_string("save_to", "output_video.mp4", "Provide a relative path for a resulting video. Default value: output_video.mp4")

# TODO: Finetune the net on wider screen and higher resolution.
flags.DEFINE_integer("resolution", 320, "Provide value for rescaling input image. Default value is 320 (320x320).")
flags.DEFINE_string("path_to_net", "logs/adv1", "Provide a relative path to a folder containing model.ckpt...... Default value: logs/adv1/model.ckpt")

FLAGS = flags.FLAGS


# Define variables for net initialization
CLASSES = ['person', 'bicycle', 'car', 'motorcycle']
coco_labels = [1, 2, 3, 4]
batch_size = 1
imshape = (FLAGS.resolution*2, FLAGS.resolution)
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
        print("Restoring a model.")
        saver = tf.train.Saver()

        # Initialize variables in the model and merge all summaries
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

    print("Creating a session.")
    sess = tf.Session(config=config, graph=graph)
    saver.restore(sess, tf.train.latest_checkpoint(path_to_net))
    print("Pipeline is ready for video processing.")

    return sess, net


def process(img, threshold=0.6, max_obj=20, target_class_index=0):


    img, _ = resize_wo_scale_dist(img, imshape)
    p = compute['net'].infer(img, compute['sess'])

    for i in range(len(p.bboxes)):

        class_indices = np.nonzero(p.classes[i]==target_class_index)[0]
        bboxes = p.bboxes[i][class_indices]
        conf = p.conf[i][class_indices]
        classes = p.classes[i][class_indices]

        label = []

        final_boxes, final_probs, final_cls_idx = \
            filter_prediction(bboxes, conf, classes,
                              PROB_THRESH=threshold, TOP_N_DETECTIONS=max_obj, n_classes=len(coco_labels))
        for box_id in range(len(final_boxes)):
            kernel_id = final_cls_idx[box_id]
            label.append(CLASSES[kernel_id] + " {}%".format(int(final_probs[box_id] * 100)))

        img = draw_boxes(img,
                         bbox_transform(np.hsplit(np.array(final_boxes),4)),
                         label, thickness=1, fontScale=0.5)

    return img

def main():

    compute['sess'], compute['net'] = define_net()

    clip = VideoFileClip(FLAGS.video)
    processed_clip = clip.fl_image(process)
    processed_clip.write_videofile(FLAGS.save_to, audio=False)


if __name__ == "__main__":
    main()










