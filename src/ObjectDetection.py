#!/usr/bin/env python
"""
SmallNet.py is a class that provides a network for Person localization
Created 6/13/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

from collections import namedtuple
import tensorflow as tf
import numpy as np
from tensorflow import variable_scope, reshape, sigmoid, reduce_mean, reduce_sum, nn,\
    unstack, identity, stack, truediv
from .util import safe_exp, bbox_transform, bbox_transform_inv, set_anchors, resize_wo_scale_dist,\
    find_anchor_ids, estimate_deltas, coco_boxes2cxcywh, coco_boxes2xmin_ymin_xmax_ymax, convertToFixedSize, sparse_to_dense, nms
from .NetTemplate import NetTemplate

Point = namedtuple('Point',['x', 'y'])
Predictions = namedtuple("Predictions", ['bboxes','conf','classes'])

class ObjectDetectionNet(NetTemplate):


    imshape, outshape = None, None



    def __init__(self, labels_provided, lr,
                 anchor_shapes=np.array([[36., 36.],[366., 174.],[115.,  59.],[78., 170.]])):
        """
        A skeleton of SqueezeDet Net.

        :param labels_provided: number of classes
        :param batch_sz: batch size
        :param gt_box: an input for ground true bounding boxes
        :param imshape: input image width(x) and height (y)
        :param anchor_shapes: anchor shapes to be applied to each cell of the feature map
        """
        self.lr = lr

        try:
            assert self.imshape is not None, "Error incorrect image input type"
            assert self.outshape is not None, "Error incorrect output input type"
        except:
            assert False, "image size error"


        self.labels_available = labels_provided
        self.n_classes = len(labels_provided)
        self.K = len(anchor_shapes)
        self.WHK = self.K * self.outshape.x * self.outshape.y
        self.anchors = set_anchors(self.imshape, self.outshape, anchor_shapes)
        self.EXP_THRESH = 1.0
        self.EPSILON = 1e-16
        self.LOSS_COEF_BBOX = 5.0 # should be float
        self.LOSS_COEF_CLASS = 1.0 # should be float
        self.LOSS_COEF_CONF_POS = 75.0 # should be float
        self.LOSS_COEF_CONF_NEG = 100.0 # should be float
        self.input_img = None

        self.input_box_values = None #[xmin, ymin, xmax, ymax]

        self.input_box_delta = None

        self.input_mask = None

        self.input_labels = None

        super().__init__()

    def setup_inputs(self, img, bbox, deltas, mask, labels):

        self.input_img = img
        tf.add_to_collection(tf.GraphKeys.RESOURCES, self.input_img)

        self.input_box_values = bbox  # [xmin, ymin, xmax, ymax]
        tf.add_to_collection(tf.GraphKeys.RESOURCES, self.input_box_values)

        self.input_box_delta = deltas
        tf.add_to_collection(tf.GraphKeys.RESOURCES, self.input_box_delta)

        self.input_mask = mask
        tf.add_to_collection(tf.GraphKeys.RESOURCES, self.input_mask)

        self.input_labels = labels
        tf.add_to_collection(tf.GraphKeys.RESOURCES, self.input_labels)

        self._add_featuremap()
        self._add_obj_detection()
        self._add_loss_graph()
        self._add_train_graph()



    def _map_to_grid(self, sparse_label, sparse_gtbox, sparse_aids, sparse_deltas):

        # Convert into a flattened out list
        label_indices, \
        bbox_indices, \
        box_delta_values, \
        mask_indices, \
        box_values = convertToFixedSize(aidx=sparse_aids,
                                        labels=sparse_label,
                                        boxes_deltas=sparse_deltas,
                                        bboxes=sparse_gtbox)

        # Extract variables to make it more readable
        n_anchors = len(self.anchors)
        n_classes = self.n_classes
        n_labels = len(label_indices)

        # Dense boxes
        label_indices = sparse_to_dense(label_indices, [n_anchors, n_classes], np.ones(n_labels, dtype=np.float))
        bbox_deltas = sparse_to_dense(bbox_indices, [n_anchors, 4], box_delta_values)
        mask = np.reshape(sparse_to_dense(mask_indices, [n_anchors], np.ones(n_labels, dtype=np.float)),
                          [n_anchors, 1])
        box_values = sparse_to_dense(bbox_indices, [n_anchors, 4], box_values)

        return {'dense_labels': label_indices,
                'masks': mask,
                'bbox_deltas': bbox_deltas,
                'bbox_values': box_values}



    def _add_featuremap(self):
        raise NotImplementedError()



    def _add_obj_detection(self):

        feature_map = self.feature_map
        n_classes = self.n_classes
        mask = self.input_mask
        b_sz = tf.shape(feature_map)[0]
        K = self.K
        KC = self.K * self.n_classes
        WHK = self.WHK
        ɛ = self.EPSILON

        with variable_scope('classes'):
            # probability
            with variable_scope('probability'):
                """
                This part carves out layers responsible for class probabilities in every box
                
                For each of the anchors, calculate class probability
                """

                class_logits = reshape(feature_map[:, :, :, :KC],
                                            [-1, n_classes])
                # Softmax calculates probability distribution over the last dimension (classes)
                P_class = nn.softmax(class_logits)
                self.P_class = reshape(P_class, [b_sz, WHK, n_classes], name="P")

            with variable_scope('confidence'):
                """
                Estimate a confidence whether an anchor overlaps with an object
                """
                # Extract anchor logits for each image (batch sz dimension)
                n_anchors_slice = (KC + K)
                anchor_confidence = reshape(feature_map[:, :, :, KC:n_anchors_slice],
                                            [b_sz, WHK])
                # Estimate highest confidence
                self.anchor_confidence = sigmoid(anchor_confidence, name="C")

        with variable_scope('box'):
            """
            """
            box_deltas = feature_map[:,:,:, n_anchors_slice:]
            self.detected_box_deltas = reshape(box_deltas, [b_sz, WHK, 4], name="deltas")
            imshape = self.imshape

            with variable_scope('stretching'):
                dx, dy, dw, dh = unstack(self.detected_box_deltas, axis=2)
                type_to_cast = dx.dtype.as_numpy_dtype
                x = np.array(self.anchors[:, 0], dtype=type_to_cast) #0 is for x
                y = np.array(self.anchors[:, 1], dtype=type_to_cast) #1 is for y
                w = np.array(self.anchors[:, 2], dtype=type_to_cast) #2 is for w
                h = np.array(self.anchors[:, 3], dtype=type_to_cast) #3 is for h

                # let's copy the result of box adjustments
                center_x = tf.add(x, dx * w, name='cx') #center_x = identity(x + dx * w, name='cx')
                center_y = tf.add(y, dy * h, name='cy') # center_y = identity(y + dy * h, name='cy')
                # exponent is used to undo the log used to pack box width and height
                width = tf.multiply(w, safe_exp(dw, self.EXP_THRESH), name='bbox_width') # width = identity(w * safe_exp(dw, self.EXP_THRESH), name='bbox_width')
                # exponent is used to undo the log used to pack box width and height
                height = tf.multiply(h, safe_exp(dh, self.EXP_THRESH), name='bbox_height') # height = identity(h * safe_exp(dh, self.EXP_THRESH), name='bbox_height')

            with variable_scope('trimming'):
                '''
                This part makes sure that the predicted values do not extend beyond the images size
                '''
                xmin, ymin, xmax, ymax = bbox_transform([center_x, center_y, width, height])
                xmin = tf.minimum(tf.maximum(0.0, xmin), np.float32(imshape.x - 1.0), name="xmin")
                xmax = tf.maximum(tf.minimum(np.float32(imshape.x - 1.0), xmax), 0.0, name="xmax")
                ymin = tf.minimum(tf.maximum(0.0, ymin), np.float32(imshape.x - 1.0), name="xmin")
                ymax = tf.maximum(tf.minimum(np.float32(imshape.y - 1.0), ymax), 0.0, name="xmax")

                det_box = stack(bbox_transform_inv([xmin, ymin, xmax, ymax]))
                self.det_boxes = tf.transpose(det_box, (1, 2, 0), name="box_prediction")
                tf.add_to_collection("predictions", self.det_boxes)


            with variable_scope('IOU'):

                box1 = bbox_transform(unstack(self.det_boxes, axis=2))
                box2 = bbox_transform(unstack(self.input_box_values, axis=2))

                with tf.variable_scope('intersection'):
                    inters_xmin = tf.maximum(box1[0], box2[0], name='xmin')
                    inters_ymin = tf.maximum(box1[1], box2[1], name='ymin')
                    inters_xmax = tf.minimum(box1[2], box2[2], name='xmax')
                    inters_ymax = tf.minimum(box1[3], box2[3], name='ymax')

                    w = tf.maximum(0.0, inters_xmax - inters_xmin, name='inter_w')
                    h = tf.maximum(0.0, inters_ymax - inters_ymin, name='inter_h')
                    intersection = tf.multiply(w, h, name='intersection')

                with tf.variable_scope('union'):
                    w1 = tf.subtract(box1[2], box1[0], name='w1')
                    h1 = tf.subtract(box1[3], box1[1], name='h1')
                    w2 = tf.subtract(box2[2], box2[0], name='w2')
                    h2 = tf.subtract(box2[3], box2[1], name='h2')

                    union = w1 * h1 + w2 * h2 - intersection

                self.IoU = tf.multiply(intersection / (union + ɛ), reshape(mask, [b_sz, self.WHK]), name='IoU')

            with variable_scope('probability'):

                probs = tf.multiply(self.P_class, reshape(self.anchor_confidence, [b_sz, WHK, 1]), name='final_class_prob')

                self.det_probs = tf.reduce_max(probs, 2, name='score')
                tf.add_to_collection("predictions", self.det_probs)
                self.det_class = tf.argmax(probs, 2, name='class_idx')
                tf.add_to_collection("predictions", self.det_class)


    def _add_loss_graph(self):
        """Define the loss operation.
        Unlike Faster R-CNN, which deploys a (4-step) alternating training strategy to train RPN and detector
        net-work, our SqueezeDet detection network can be trained end-to-end, similarly to YOLO.

        To train the ConvDet layer to learn detection, localization and classification, we define a multi-task
        loss function:

        W_bbox * bbox_loss + C_loss + W_ce * P_loss

        Where:

        bbox_loss = mask * ( (dx' - dx)^2 + (dy' - dy)^2 + (dw'-dw)^2 + (dh'-dh)^2 )
            -   mask is 1 if anchor and ground truth input_mask is this highest and 0 otherwise
                dx, dy, dw, dh - distance between ground truth and anchors center, width and height
                dx', dy', dw', dh' - model estimates

        C_loss = W_pos/N_obj * mask * (confidence' - 1)^2 + W_neg/(WHK-N_obj) * (1-mask) * confidence'^2
            -   W_pos/N_obj and W_neg/(WHK-N_obj) are normalized weights of the loss

        P_loss = mask * OHE(label)*log(Pc) / N_obj

        Reference:
        Wu, B., Iandola, F., Jin, P. H., Keutzer, K., Huang, J., Rathod, V., … Research, G. (2016).
        SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection
        for Autonomous Driving. arXiv.
        """

        b_sz = tf.shape(self.feature_map)[0]
        mask = self.input_mask # is 1 if anchor and ground truth input_mask is this highest and 0 otherwise
        W_bbox = self.LOSS_COEF_BBOX # weigt of bounding box loss
        W_ce = self.LOSS_COEF_CLASS # weight of cross entropy  loss (P(class))
        W_pos = self.LOSS_COEF_CONF_POS # weight of confidence loss in positive examples
        W_neg = self.LOSS_COEF_CONF_NEG # weight of confidence loss in negative examples
        n_obj = reduce_sum(mask) # number of target objects in an image - used to normalize loss
        tf.summary.scalar("num_objects", n_obj)
        WHK = self.WHK # feature map width * height * K anchors per cell
        L = self.input_labels
        ɛ = self.EPSILON

        # If batch norm is used it is important to add dependency to update UPDATE_OPS before you do loss calc
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            with variable_scope('Loss'):
                with variable_scope('P_class'):

                    # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
                    # add a small value into log to prevent blowing up
                    pos_CE = L * -tf.log(self.P_class + ɛ)
                    neg_CE = (1-L) * -tf.log(1 - self.P_class + ɛ)
                    self.P_loss = truediv(W_ce * reduce_sum((pos_CE + neg_CE) * mask), n_obj, name='loss')
                    # add to a collection called losses to sum those losses later
                    tf.add_to_collection(tf.GraphKeys.LOSSES, self.P_loss)

                with tf.variable_scope('Confidence'):

                    anchor_mask = reshape(mask, [b_sz, WHK])

                    conf_pos = tf.square(self.IoU - self.anchor_confidence)
                    norm = (anchor_mask * W_pos / n_obj + (1 - anchor_mask ) * W_neg / (WHK - n_obj))

                    self.conf_loss = reduce_mean(reduce_sum(conf_pos * norm, 1), name='loss')

                    tf.add_to_collection(tf.GraphKeys.LOSSES, self.conf_loss)

                with tf.variable_scope('BBox'):

                    all_deltas = self.detected_box_deltas - self.input_box_delta
                    pos_deltas = all_deltas * mask
                    norm_deltas = reduce_sum(W_bbox * tf.square(pos_deltas))
                    norm_deltas = truediv(norm_deltas, n_obj)

                    self.bbox_loss = tf.multiply(W_bbox, norm_deltas, name='loss')

                    tf.add_to_collection(tf.GraphKeys.LOSSES, self.bbox_loss)

                # add above losses as well as weight decay losses to form the total loss
                self.loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')


    def _add_train_graph(self):
        """Define the training operation."""
        self._add_loss_summaries(self.loss)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = opt.minimize(self.loss)
        tf.add_to_collection(tf.GraphKeys.TRAIN_OP, self.train_op)

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(var.op.name, var)


    def _add_loss_summaries(self, total_loss):
        """
        Add summaries for losses

        Generates loss summaries for visualizing the performance of the network.

        Args:
          total_loss: Total loss from loss().
        """
        losses = tf.get_collection(tf.GraphKeys.LOSSES)


        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name, l)


    def infer(self, X_batch, sess):
        p = Predictions(*sess.run([self.det_boxes, self.det_probs, self.det_class],
                                  feed_dict={self.input_img: np.expand_dims(X_batch, 0),
                                             self.is_training: False}))
        return p


    def filter_prediction(self, boxes, probs, cls_idx, TOP_N_DETECTION=50, PROB_THRESH=0.5, NMS_THRESH=0.2):
        """Filter bounding box predictions with probability threshold and
        non-maximum supression.
        Args:
          boxes: array of [cx, cy, w, h].
          probs: array of probabilities
          cls_idx: array of class indices
        Returns:
          final_boxes: array of filtered bounding boxes.
          final_probs: array of filtered probabilities
          final_cls_idx: array of filtered class indices
        """

        if TOP_N_DETECTION < len(probs) and TOP_N_DETECTION > 0:
            order = probs.argsort()[:-TOP_N_DETECTION - 1:-1]
            probs = probs[order]
            boxes = boxes[order]
            cls_idx = cls_idx[order]
        else:
            filtered_idx = np.nonzero(probs > PROB_THRESH)[0]
            probs = probs[filtered_idx]
            boxes = boxes[filtered_idx]
            cls_idx = cls_idx[filtered_idx]

        final_boxes = []
        final_probs = []
        final_cls_idx = []

        for c in range(self.n_classes):
            idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
            keep = nms(boxes[idx_per_class], probs[idx_per_class], NMS_THRESH)
            for i in range(len(keep)):
                if keep[i]:
                    final_boxes.append(boxes[idx_per_class[i]])
                    final_probs.append(probs[idx_per_class[i]])
                    final_cls_idx.append(c)

        return final_boxes, final_probs, final_cls_idx


    def preprocess_COCO(self, img, labels, bboxes):
        """
        Transforms inputs into expected net format.
        :param img: RGB img -> will be scaled to a fixed size
        :param labels:
        :param bboxes: [[cx,cy,w,h]] where y=0 is the image bottom - will be scalled to a fixed size
        :return: img, labels, mask, bbox_values, bbox_deltas
        """

        # 1. Rescale images to the same fixed size
        im, scale = resize_wo_scale_dist(img, self.imshape)
        bboxes = np.array(bboxes) * scale

        # 2. Convert COCO bounding boxes into cx, cy, w, h format and labels into net native format
        bboxes = list(map(lambda x: coco_boxes2cxcywh(im, x), bboxes))
        labels = list(map(lambda x: self.labels_available.index(x), labels))

        # 3. Find mask (anchor ids)
        aids = find_anchor_ids(bboxes, self.anchors)

        # 4. Calculate deltas between anchors and bounding boxes
        deltas = estimate_deltas(bboxes, aids, self.anchors)

        # 5. Convert to dense annotations

        dense = self._map_to_grid(labels, bboxes, aids, deltas)

        return im, dense['dense_labels'], dense['masks'], dense['bbox_deltas'], dense['bbox_values']






