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
from tensorflow import placeholder, FIFOQueue
from tensorflow.python.platform.app import flags
from tensorflow.python import debug as tf_debg
from src.COCO_DB import COCO
from src.SmallNet import SmallNet as Net

CLASSES = ['person', 'bicycle', 'car', 'motorcycle']
ANNOTATIONS_FILE = 'dataset/annotations/instances_train2014.json'
PATH2IMAGES = 'dataset/images/train2014'
coco_labels=[1, 2, 3, 4]


FLAGS = flags.FLAGS
flags.DEFINE_string("train_dir", "logs/t1", "Provide training directory for recovering and storing model. Default value is logs/t6")
flags.DEFINE_float("learning_rate", 1e-3, "Provide a value for learning rate. Default value is 1e-3")
flags.DEFINE_bool("restore", False, "Do you want to restore a model? Default value is False.")
flags.DEFINE_integer("batch_size", 128, "Provide a size of the minibatch. Default value is 128.")
flags.DEFINE_integer("resolution", 320, "Provide value for rescaling input image. Default value is 320 (320x320).")
flags.DEFINE_bool("debug", True, "Set to True to enter into a debugging mode. Default value is False.")
flags.DEFINE_string("activations", 'elu', "Set activations. Default type is elu. Available options are elu, relu")
flags.DEFINE_string("optimizer", "adam", "Set optimization algorithm. Default value is adam. Available options are [adam, rmsprop, momentum].")

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




coco = COCO(ANNOTATIONS_FILE, PATH2IMAGES, CLASSES)

def generate_sample(net):
    looking = True
    if FLAGS.debug:

        ################
        ### Debuggin ###
        ################

        im, bboxes, deltas, mask, labels = [],[],[],[],[]
        while looking:
            try:
                im_, l_, b_ = coco.get_sample()
                im_, l_, m_, d_, b_ = net.preprocess_COCO(im_, l_, b_)
                im.append(im_)
                bboxes.append(bboxes)
                deltas.append(d_)
                mask.append(m_)
                labels.append(l_)
                looking = False if len(im)>=batch_sz else True
            except:
                pass

    else:
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
    with graph.as_default():
        with tf.device("gpu:{}".format(gpu_id)):
            net = Net(coco_labels, imshape, learning_rate, activations=FLAGS.activations, width=0.5)

            # Create inputs
            if FLAGS.debug:

                ################
                ### Debuggin ###
                ################
                im_ph = placeholder(dtype=tf.float32, shape=[batch_sz, imshape[0], imshape[1], 3], name="img")
                labels_ph = placeholder(dtype=tf.float32, shape=[batch_sz, net.WHK, net.n_classes], name="labels")
                mask_ph = placeholder(dtype=tf.float32, shape=[batch_sz, net.WHK, 1], name="mask")
                deltas_ph = placeholder(dtype=tf.float32, shape=[batch_sz, net.WHK, 4], name="deltas_gt")
                bbox_ph = placeholder(dtype=tf.float32, shape=[batch_sz, net.WHK, 4], name="bbox_gt")

            else:
                im_ph = placeholder(dtype=tf.float32,shape=[*imshape[::-1], 3],name="img")
                labels_ph = placeholder(dtype=tf.float32, shape=[net.WHK, net.n_classes], name="labels")
                mask_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 1], name="mask")
                deltas_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 4], name="deltas_gt")
                bbox_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 4], name="bbox_gt")

            inputs = (im_ph, bbox_ph, deltas_ph, mask_ph, labels_ph)

            if FLAGS.debug:

                ################
                ### Debuggin ###
                ################

                pass

            else:

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
                # Launch coordinator that will manage threads
                coord = tf.train.Coordinator()

            net.optimization_op = FLAGS.optimizer
            if FLAGS.debug:

                ################
                ### Debuggin ###
                ################
                net.setup_inputs(im_ph, bbox_ph, deltas_ph, mask_ph, labels_ph)
            else:
                net.setup_inputs(*dequeue_op)

        # Initialize variables in the model and merge all summaries
        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(train_dir, graph)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config, graph=graph)
    sess.run(initializer)

    if FLAGS.debug:
        sess = tf_debg.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debg.has_inf_or_nan)

    if restore_model:
        saver = tf.train.Saver(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.restore(sess, join(train_dir, 'model.ckpt'))


    if not FLAGS.debug:
        tf.train.start_queue_runners(sess=sess, coord=coord)
        threads = [threading.Thread(target=enqueue_thread, args=(coord, sess, net, enqueue_op, inputs)).start()
                   for _ in range(prefetching_threads)]

    pass_tracker_start = time.time()
    pass_tracker_prior = pass_tracker_start
    pbar = trange(max_steps)

    prior_step = 0
    print("Beginning training process.")

    for step in pbar:

        # Configure operation that TF should run depending on the step number
        if step % summary_step == 0:
            op_list = [net.train_op, net.loss, summary_op, net.P_loss, net.conf_loss, net.bbox_loss]

            if FLAGS.debug:

                ################
                ### Debuggin ###
                ################
                print("Generate data")
                data = generate_sample(net)
                print("Finished data generation")
                _, loss_value, summary_str, class_loss, conf_loss, bbox_loss = \
                    sess.run(op_list, feed_dict = {im_ph: data[0], bbox_ph: data[1], deltas_ph: data[2],
                                                   mask_ph: data[3], labels_ph: data[4], net.is_training: False})
                print("Processed one batch")
            else:
                _, loss_value, summary_str, class_loss, conf_loss, bbox_loss = sess.run(op_list, feed_dict={net.is_training: False})

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

        else:

            if FLAGS.debug:

                ################
                ### Debuggin ###
                ################

                print("Generate data")
                data = generate_sample(net)
                print("Finished data generation")
                _, loss_value, conf_loss, bbox_loss, class_loss = \
                    sess.run([net.train_op, net.loss, net.conf_loss, net.bbox_loss, net.P_loss],
                             feed_dict = {im_ph: data[0], bbox_ph: data[1], deltas_ph: data[2],
                                          mask_ph: data[3], labels_ph: data[4], net.is_training: True})
                print("Processed one batch")
            else:
                _, loss_value, conf_loss, bbox_loss, class_loss = \
                    sess.run([net.train_op, net.loss, net.conf_loss, net.bbox_loss, net.P_loss],
                             feed_dict={net.is_training: True}) # , feed_dict={net.dropout_rate: 0.75}

            pbar.set_postfix(bbox_loss="{:.1f}".format(bbox_loss),
                             class_loss="{:.1f}%".format(class_loss*100),
                             total_loss="{:.2f}".format(loss_value))

        assert not np.isnan(loss_value), \
            'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
            'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

        # Save the model checkpoint periodically.
        if step % checkpoint_step == 0 or step == max_steps:
            summary_writer.add_summary(summary_str, step)
            checkpoint_path = join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path)

    # Close a queue and cancel all elements in the queue. Request coordinator to stop all the threads.
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    # Tell coordinator to stop any queries to the threads
    coord.join(threads)
    sess.close()


if __name__=='__main__':
    train()





