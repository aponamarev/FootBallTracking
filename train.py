#!/usr/bin/env python
"""
train.py is a training pipeline for training object detection
Created 6/15/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import tensorflow as tf
import threading
import numpy as np
import time
from os.path import join
from tqdm import trange
from tensorflow import placeholder, RandomShuffleQueue
from tensorflow.python.platform.app import flags
from src.COCO_DB import COCO
from src.SmallNet import SmallNet
from src.AdvancedNet import AdvancedNet

CLASSES = ['person', 'bicycle', 'car', 'motorcycle']
ANNOTATIONS_FILE = 'dataset/annotations/instances_train2014.json'
PATH2IMAGES = 'dataset/images/train2014'
coco_labels=[1, 2, 3, 4]


FLAGS = flags.FLAGS
flags.DEFINE_string("train_dir", "logs/small", "Provide training directory for recovering and storing model. Default value is logs/t6")
flags.DEFINE_float("learning_rate", 1e-3, "Provide a value for learning rate. Default value is 1e-3")
flags.DEFINE_bool("restore", True, "Do you want to restore a model? Default value is True.")
flags.DEFINE_integer("batch_size", 128, "Provide a size of the minibatch. Default value is 128.")
flags.DEFINE_integer("resolution", 320, "Provide value for rescaling input image. Default value is 320 (320x320).")
flags.DEFINE_bool("debug", False, "Set to True to enter into a debugging mode. Default value is False.")
flags.DEFINE_string("activations", 'elu', "Set activations. Default type is elu. Available options are elu, relu")
flags.DEFINE_string("optimizer", "adam", "Set optimization algorithm. Default value is adam. Available options are [adam, rmsprop, momentum].")
flags.DEFINE_string("net", "small", "Set a net. Default SmallNet. Options are [small, advanced]")
flags.DEFINE_float("width", 0.5, "Set the net width multiple. Default is 1.0. Type Float")

print("Starting training process")
print("Model restore/save folder", FLAGS.train_dir)
print("Model restore status", FLAGS.restore)
print("Model net", FLAGS.net)

train_dir = FLAGS.train_dir
learning_rate = FLAGS.learning_rate
restore_model = FLAGS.restore
batch_sz=FLAGS.batch_size
imshape=(FLAGS.resolution*2, FLAGS.resolution)

queue_capacity = batch_sz * 5
prefetching_threads = 4
gpu_id = 0

summary_step = 100
checkpoint_step = 1000
max_steps = 10**6

Net = {'small': SmallNet, 'advanced': AdvancedNet}
assert FLAGS.net in Net.keys(), "Incorrect net key provided. Expected keys: {}. Provided key {}".\
    format(list(Net.keys()), FLAGS.net)
Net = Net[FLAGS.net]



coco = COCO(ANNOTATIONS_FILE, PATH2IMAGES, CLASSES)

def generate_sample(net):
    looking = True
    while looking:
        try:
            im, labels, bboxes = coco.get_sample()
            im, labels, mask, deltas, bboxes = net.preprocess_COCO(im, labels, bboxes)
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
    with graph.as_default() as g:
        with tf.device("gpu:{}".format(gpu_id)):
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
            queue = RandomShuffleQueue(capacity=queue_capacity, min_after_dequeue= 2*batch_sz,
                                       dtypes=[v.dtype for v in inputs], shapes=shapes)
            enqueue_op = queue.enqueue(inputs)
            dequeue_op = tf.train.batch(queue.dequeue(), batch_sz, num_threads=prefetching_threads,
                                        capacity=queue_capacity, shapes=shapes, name="Batch_{}_samples".format(batch_sz))

            net.optimization_op = FLAGS.optimizer
            print("Building {}.".format(type(net).__name__))
            net.setup_inputs(*dequeue_op)
            net.debug = FLAGS.debug

        # Initialize variables in the model and merge all summaries
        initializer = tf.global_variables_initializer()
        restore_variables = tf.global_variables()
        saver = tf.train.Saver(restore_variables, reshape=True)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(train_dir, g)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        print("Creating a session.")
        sess = tf.Session(config=config, graph=g)
        print("Initializing/restoring variables.")
        if restore_model:
            saver.restore(sess, join(train_dir, 'model.ckpt'))
        else:
            sess.run(initializer)
    print("Start data prefetching")
    # Launch coordinator that will manage threads
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    threads = [threading.Thread(target=enqueue_thread, args=(coord, sess, net, enqueue_op, inputs)).start()
     for _ in range(prefetching_threads)]

    pass_tracker_start = time.time()
    pass_tracker_prior = pass_tracker_start
    pbar = trange(max_steps)
    print("Pipeline built is completed successfully.")
    prior_step = 0
    print("Beginning training process.")

    for step in pbar:

        # Configure operation that TF should run depending on the step number
        if step % summary_step == summary_step-1:
            op_list = [net.train_op, net.loss, summary_op, net.P_loss, net.conf_loss, net.bbox_loss]

            _, loss_value, summary_str, class_loss, conf_loss, bbox_loss = sess.run(op_list,
                                                                                    feed_dict={net.is_training: False,
                                                                                               net.keep_prob: 1.0})

            pass_tracker_end = time.time()

            summary_writer.add_summary(summary_str, step)

            # Report results
            number_of_steps = step - prior_step
            number_of_steps = number_of_steps if number_of_steps > 0 else 1
            print('\nStep: {}. Timer: {} network passes (with batch size {}): {:.1f} seconds ({:.1f} per batch). Losses: conf_loss: {:.3f}, bbox_loss: {:.3f}, class_loss: {:.3f} and total_loss: {:.3f}'.
                  format(step, number_of_steps, batch_sz,
                         pass_tracker_end - pass_tracker_prior,
                         (pass_tracker_end - pass_tracker_prior) / number_of_steps,
                         conf_loss, bbox_loss, class_loss, loss_value))
            pass_tracker_prior = pass_tracker_end
            prior_step = step

        # Save the model checkpoint periodically.
        elif step % checkpoint_step == 0 or step == max_steps:
            checkpoint_path = join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path)

        else:

            _, loss_value, conf_loss, bbox_loss, class_loss = \
                sess.run([net.train_op, net.loss, net.conf_loss, net.bbox_loss, net.P_loss], feed_dict={net.is_training: True,
                                                                                                        net.keep_prob: 0.75})

            pbar.set_postfix(bbox_loss="{:.1f}".format(bbox_loss),
                             class_loss="{:.1f}%".format(class_loss),
                             total_loss="{:.2f}".format(loss_value))

            assert not np.isnan(loss_value), \
                'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
                'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

    # Close a queue and cancel all elements in the queue. Request coordinator to stop all the threads.
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    # Tell coordinator to stop any queries to the threads
    coord.join(threads)
    sess.close()


if __name__=='__main__':
    train()





