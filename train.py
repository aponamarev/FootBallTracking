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
from src.COCO_DB import COCO
from src.SmallNet import SmallNet
from src.util import draw_boxes, coco_boxes2xmin_ymin_xmax_ymax, \
    coco_boxes2cxcywh, cxcywh_xmin_ymin_xmax_ymax, resize_wo_scale_dist

CLASSES = ['person']
ANNOTATIONS_FILE = 'src/coco/annotations/instances_train2014.json'
PATH2IMAGES = 'src/coco/images/train2014'
train_dir = 'logs/t1'

coco_labels=[1]
batch_sz=16
queue_capacity = batch_sz * 12
prefetching_threads = 2
imshape=(768, 512)
gpu_id = 0

summary_step = 20
checkpoint_step = 1000
max_steps = 10**6

coco = COCO(ANNOTATIONS_FILE, PATH2IMAGES, CLASSES) # TODO: Move to after the set up of inputs to speed up debugging

def generate_sample(net):
    looking = True
    while looking:
        try:
            im, labels, bboxes = coco.get_sample()
            im, labels, mask, deltas, bboxes = net.preprocess_COCO(im, labels, bboxes)
            looking = False
        except:
            print("Caught an error in image processing")

    return [im, bboxes, deltas, mask, labels]


def enqueue_thread(coord, sess, net, enqueue_op, inputs):
    with coord.stop_on_exception():
        while not coord.should_stop():
            data = generate_sample(net)
            print(".", sep="", end=" ", flush=True)
            sess.run(enqueue_op, feed_dict={ph:v for ph,v in zip(inputs, data)})



def train():

    graph = tf.Graph()
    with graph.as_default():
        with tf.device("gpu:{}".format(gpu_id)):
            net = SmallNet(coco_labels, batch_sz, imshape)
            im_ph = placeholder(dtype=tf.float32,shape=[*imshape[::-1], 3],name="img")
            labels_ph = placeholder(dtype=tf.float32, shape=[net.WHK, net.n_classes], name="labels")
            mask_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 1], name="mask")
            deltas_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 4], name="deltas_gt")
            bbox_ph = placeholder(dtype=tf.float32, shape=[net.WHK, 4], name="bbox_gt")
            # Add inputs to graph collections to have access to them after recovery.
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
                                        shapes=shapes, name="Batch_{}_samples".format(batch_sz), num_threads=prefetching_threads)
            net.setup_inputs(*dequeue_op)

        # Initialize variables in the model and merge all summaries
        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(train_dir, graph)

        # Launch coordinator that will manage threads
        coord = tf.train.Coordinator()
        graph.finalize()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config, graph=graph)
    sess.run(initializer)

    tf.train.start_queue_runners(sess=sess, coord=coord)

    threads = [threading.Thread(target=enqueue_thread, args=(coord, sess, net, enqueue_op, inputs)).start()
               for _ in range(prefetching_threads)]

    pass_tracker_start = time.time()
    pass_tracker_prior = pass_tracker_start
    pbar = range(max_steps)

    prior_step = 0
    print("Beginning training process.")

    for step in pbar:

        # Configure operation that TF should run depending on the step number
        if step % summary_step == 0:
            op_list = [
                net.train_op,
                net.loss,
                summary_op,
                net.P_loss,
                net.conf_loss,
                net.bbox_loss,
            ]

            _, loss_value, summary_str, class_loss, conf_loss, bbox_loss = \
                sess.run(op_list, feed_dict={net.is_training_mode: False}) # , feed_dict={net.dropout_keep_rate: 1.0}

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
            _, loss_value, conf_loss, bbox_loss, class_loss = \
                sess.run([net.train_op, net.loss, net.conf_loss, net.bbox_loss, net.P_loss],
                         feed_dict={net.is_training_mode: True}) # , feed_dict={net.dropout_keep_rate: 0.75}

        assert not np.isnan(loss_value), \
            'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
            'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

        # Save the model checkpoint periodically.
        if step % checkpoint_step == 0 or (step + 1) == max_steps-1:
            #viz_summary = sess.run(net.viz_op)
            summary_writer.add_summary(summary_str, step)
            #summary_writer.add_summary(viz_summary, step)
            checkpoint_path = join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path)

        #threads = [threading.Thread(target=enqueue_thread, args=(coord,)).start() for i in range(prefetching_threads)]

        # Close a queue and cancel all elements in the queue. Request coordinator to stop all the threads.
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    # Tell coordinator to stop any queries to the threads
    coord.join(threads)
    sess.close()


if __name__=='__main__':
    train()





