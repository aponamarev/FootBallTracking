#!/usr/bin/env python
"""
train.py is a training pipeline for training object detection
Created 6/15/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import tensorflow as tf
import numpy as np
import threading, os
from matplotlib import pyplot as plt
from tensorflow import placeholder, FIFOQueue
from tensorflow.python.platform.app import flags
from src.COCO_DB import COCO
from src.AdvancedNet import AdvancedNet as Net
from src.AdvancedNet import AdvancedNet
from src.util import filter_prediction, bbox_transform, draw_boxes

CLASSES = ['person', 'bicycle', 'car', 'motorcycle']
ANNOTATIONS_FILE = 'dataset/annotations/instances_train2014.json'
PATH2IMAGES = 'dataset/images/train2014'
coco_labels=[1, 2, 3, 4]


FLAGS = flags.FLAGS
flags.DEFINE_string("train_dir", "logs/adv1", "Provide training directory for recovering and storing model. Default value is logs/t6")
flags.DEFINE_float("learning_rate", 1e-3, "Provide a value for learning rate. Default value is 1e-3")
flags.DEFINE_bool("restore", True, "Do you want to restore a model? Default value is False.")
flags.DEFINE_integer("batch_size", 128, "Provide a size of the minibatch. Default value is 128.")
flags.DEFINE_integer("resolution", 320, "Provide value for rescaling input image. Default value is 320 (320x320).")
flags.DEFINE_bool("debug", False, "Set to True to enter into a debugging mode. Default value is False.")
flags.DEFINE_string("activations", 'elu', "Set activations. Default type is elu. Available options are elu, relu")
flags.DEFINE_string("optimizer", "adam", "Set optimization algorithm. Default value is adam. Available options are [adam, rmsprop, momentum].")
flags.DEFINE_string("net", "small", "Set a net. Default SmallNet. Options are [small, advanced]")
flags.DEFINE_float("width", 0.5, "Set the net width multiple. Default is 1.0. Type Float")

train_dir = FLAGS.train_dir
learning_rate = FLAGS.learning_rate
restore_model = FLAGS.restore
batch_sz=FLAGS.batch_size
imshape=(FLAGS.resolution, FLAGS.resolution)

queue_capacity = batch_sz * 4
prefetching_threads = 2
gpu_id = 0

summary_step = 100
checkpoint_step = 1000
max_steps = 10**6

Net = {'small': Net, 'advanced': AdvancedNet}
Net = Net[FLAGS.net]



coco = COCO(ANNOTATIONS_FILE, PATH2IMAGES, CLASSES)

def generate_sample(net):
    looking = True
    while looking:
        try:
            im, labels, bboxes = coco.get_sample()
            im, labels, mask, deltas, bboxes = net.preprocess_COCO(im, labels, bboxes)
            assert len(mask)>0, "Invald sample - no mask."
            assert len(labels)>0, "Invald sample - no labels."
            assert len(deltas)>0, "Invald sample - no deltas."
            assert len(bboxes)>0, "Invald sample - no bboxes."
            looking = False
        except:
            pass

    return [im, bboxes, deltas, mask, labels]


def enqueue_thread(coord, sess, net, enqueue_op, inputs):
    with coord.stop_on_exception():
        while not coord.should_stop():
            data = generate_sample(net)
            sess.run(enqueue_op, feed_dict={ph:v for ph,v in zip(inputs, data)})



def train():

    graph = tf.Graph()
    with graph.as_default():
        net = Net(coco_labels, imshape, learning_rate, activations=FLAGS.activations, width=FLAGS.width)

        # Create inputs
        im_ph = placeholder(dtype=tf.float32,shape=[*imshape[::-1], 3],name="img")
        labels_ph = placeholder(dtype=tf.float32, shape=[net.WHK, net.n_classes], name="labels")
        mask_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 1], name="mask")
        deltas_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 4], name="deltas_gt")
        bbox_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 4], name="bbox_gt")

        inputs = (im_ph, bbox_ph, deltas_ph, mask_ph, labels_ph)

        # Create a queue that will be prefetching samples
        shapes = [v.get_shape().as_list() for v in inputs]
        queue = FIFOQueue(capacity=queue_capacity,
                          dtypes=[v.dtype for v in inputs],
                          shapes=shapes)
        # It is interesting to monitor the size of the buffer
        q_size = queue.size()
        tf.summary.scalar("prefetching_queue_size", q_size)
        enqueue_op = queue.enqueue(inputs)
        dequeue_op = tf.train.batch(queue.dequeue(), batch_sz, capacity=int(queue_capacity),
                                    shapes=shapes, name="Batch_{}_samples".format(batch_sz),
                                    num_threads=prefetching_threads)

        net.optimization_op = FLAGS.optimizer
        net.setup_inputs(*dequeue_op)
        # Launch coordinator that will manage threads
        coord = tf.train.Coordinator()

        # Initialize variables in the model and merge all summaries
        initializer = tf.global_variables_initializer()

    sess = tf.Session(graph=graph)
    sess.run(initializer)

    #restore_variables = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #saver = tf.train.Saver(restore_variables, reshape=True)
    #saver.restore(sess, os.path.join(train_dir, 'model.ckpt'))

    tf.train.start_queue_runners(sess=sess, coord=coord)
    threads = [threading.Thread(target=enqueue_thread, args=(coord, sess, net, enqueue_op, inputs)).start()
               for _ in range(prefetching_threads)]

    # Configure operation that TF should run depending on the step number
    op_list = [net.input_img, net.input_mask, net.input_box_delta, net.input_box_values, net.input_labels]

    for step in range(10):

        ims, masks, deltas, bboxes, labels = sess.run(op_list, feed_dict={net.is_training: True})

        for i in range(len(ims)):

            label = []

            final_boxes, final_probs, final_cls_idx, anch_ids = filter_prediction(bboxes[i], masks[i], labels[i],
                                                                                  PROB_THRESH=0.5, TOP_N_DETECTIONS=50)

            for box_id in range(len(final_boxes)):
                kernel_id = int(np.argmax(final_cls_idx[box_id]))
                label.append(CLASSES[kernel_id] + " {}%".format(int(final_probs[box_id] * 100)))

            img = draw_boxes(ims[i], list(map(lambda x: bbox_transform(x), final_boxes)), label,
                             thickness=1, fontScale=0.5)

            plt.imshow(img)


    # Close a queue and cancel all elements in the queue. Request coordinator to stop all the threads.
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    # Tell coordinator to stop any queries to the threads
    coord.join(threads)
    sess.close()


if __name__=='__main__':
    train()





