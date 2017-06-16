#!/usr/bin/env python
"""
util.py contains support functions used in object detection
Created 6/14/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import numpy as np
import tensorflow as tf
from collections import namedtuple
from os.path import join, exists
from cv2 import rectangle, putText, resize, getTextSize, FONT_HERSHEY_COMPLEX_SMALL


Point = namedtuple('Point',['x', 'y'])

def safe_exp(w, thresh):
    """Safe exponential function for tensors."""
    slope = np.exp(thresh)
    with tf.variable_scope('safe_exponential'):
        lin_region = tf.to_float(w > thresh)

        lin_out = slope*(w - thresh + 1.)
        exp_out = tf.exp(w)

        out = lin_region*lin_out + (1.-lin_region)*exp_out

    return out

def bbox_transform(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    with tf.variable_scope('bbox_transform') as scope:
        cx, cy, w, h = bbox
        out_box = [[]]*4
        out_box[0] = cx-w/2
        out_box[1] = cy-h/2
        out_box[2] = cx+w/2
        out_box[3] = cy+h/2

    return out_box

def set_anchors(input_shape, output_shape,
                anchor_shapes=np.array([[36., 36.], [366., 174.], [115., 59.], [78., 170.]])):
    """

    :param input_shape: Input image shape (x,y)
    :param output_shape: Output shape (x', y')
    :param anchor_shapes: anchor shapes [[w,h]...]
    :return: [x' * y' * K, [cx, cy, w, h]]
    """
    input_shape = Point(*input_shape)
    output_shape = Point(*output_shape)

    W, H, K = output_shape.x, output_shape.y, len(anchor_shapes)
    Wi, Hi = input_shape

    anchor_shapes = np.reshape([anchor_shapes] * W * H, (H, W, K, 2))

    center_x = np.arange(1, W + 1) * float(Wi) / W
    center_x = np.reshape(np.transpose(np.reshape(np.array([center_x] * H * K), (K, H, W)),
                                       (1, 2, 0)), (H, W, K, 1))
    center_y = np.arange(1, H + 1) * float(Hi) / H
    center_y = np.reshape(
        np.transpose(np.reshape(np.array([center_y] * W * K), (K, W, H)),
                     (2, 1, 0)), (H, W, K, 1))

    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )

    return anchors


def check_path(p):
    """
    Convenience method to check whether a provides path (p) is valid.
    :param p: path in question
    :return: path
    """

    if not exists(p):
        raise FileExistsError("Provided path doesn't exist: {}".format_map(p))

    return p

def draw_boxes(img, boxes_xmin_ymin_xmax_ymax, labels, thickness=2, fontFace=FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.75, color=(255, 0, 0)):
    """
    Draws bounding boxes_xmin_ymin_xmax_ymax and labels on an image
    :param img:
    :param boxes_xmin_ymin_xmax_ymax: y=0 is the bottom of the image
    :param labels:
    :param thickness:
    :param fontFace:
    :param fontScale:
    :param color:
    :return: img
    """

    Y = img.shape[0]

    for b,l in zip(boxes_xmin_ymin_xmax_ymax, labels):

        xmin, ymin, xmax, ymax = b

        top_left = (int(xmin), int(Y-ymax))
        bottom_right = (int(xmax), int(Y-ymin))

        rectangle(img, top_left, bottom_right,color, thickness=thickness)
        (w, h), baseline = getTextSize(l, fontFace, fontScale, max(1, int(thickness / 2)))
        label_y = ymax + thickness*2 + 1
        if label_y + h > Y:
            label_y = ymin - thickness*2 -1 - h
            label_y = max((2, label_y))
        label_x = (xmin+xmax-w)/2
        label_x = max((1, label_x))

        origin = (int(label_x), int(Y-label_y))
        putText(img, l, origin, fontFace, fontScale, color, thickness)


def coco_boxes2xmin_ymin_xmax_ymax(img, box):
    """
    Converts coco native format into xmin, ymin, xmax, ymax
    :param img:
    :param box: [x,y,width,height] top left x,y and width, height delta to get to bottom right
    :return:
    """
    Y = img.shape[0]
    xmin, ymax, w, h = box
    ymax = Y-ymax
    xmax = xmin+w
    ymin = ymax-h

    return int(xmin), int(ymin), int(xmax), int(ymax)


def cxcywh_xmin_ymin_xmax_ymax(box):
    """
    Convert coordinates from cx, cy, w, h format into xmin ymin xmax ymax
    :param box: cx, cy, w, h
    :return: xmin ymin xmax ymax
    """

    x,y,w,h = box

    xmin = x-w/2
    xmax = x+w/2
    ymin = y - h / 2
    ymax = y + h / 2

    return int(xmin), int(ymin), int(xmax), int(ymax)


def coco_boxes2cxcywh(img, box):
    """
    Converts coco native format into xmin, ymin, xmax, ymax
    :param img:
    :param box: [x,y,width,height] top left x,y and width, height delta to get to bottom right
    :return:
    """
    Y = img.shape[0]
    xmin, ymax, w, h = box
    ymax = Y-ymax
    x = xmin + w/2
    y = ymax - h/2

    return int(x), int(y), int(w), int(h)


def resize_wo_scale_dist(img, shape):
    """
    resize image to a fixed size without scale distortion. Missing areas will be padded with zeros.
    :param img: RGB image
    :param shape: width, height
    :return: img
    """

    W, H = shape
    h,w,c = img.shape

    mask = np.zeros((H,W,c), dtype=np.uint8)

    if h<w:
        scale = W / w
        H = h * scale
    else:
        scale = H / h
        W = w * scale

    W, H = int(W),int(H)
    img = resize(img, (W, H))

    mask[:H, :W] = img

    return mask, scale

def find_anchor_ids(bboxes, anchors):
    """
    Identifies anchor ids_per_img responsible for object detection
    :param bboxes:
    :return: anchor ids_per_img
    """
    ids_per_img = []
    id_iterator = set()
    aid = len(anchors)
    for box in bboxes:
        overlaps = batch_iou(anchors, box)
        for id in np.argsort(overlaps)[::-1]:
            if overlaps[id] <= 0:
                break
            if id not in id_iterator:
                id_iterator.add(id)
                aid = id
                break
        ids_per_img.append(aid)
    return ids_per_img


def estimate_deltas(bboxes, anchor_ids, anchors):
    """Calculates the deltas of ANCHOR_BOX and ground truth boxes.
    :param bboxes: an array of ground trueth bounding boxes (bboxes) for an image [center_x, center_y,
    width, height]
    :param anchor_ids: ids per each ground truth box that have the highest IOU
    :return: [anchor_center_x_delta,anchor_center_y_delta, log(anchor_width_scale), log(anchor_height_scale)]
    """
    assert len(bboxes)==len(anchor_ids),\
        "Incorrect arrays provided for bboxes (len[{}]) and aids (len[{}]).".format(len(bboxes), len(anchor_ids)) +\
        " Provided arrays should have the same length. "
    delta_per_img = []
    for box, aid in zip(bboxes, anchor_ids):
        # calculate deltas
        # unpack the box
        box_cx, box_cy, box_w, box_h = box
        # initialize a delta array [x,y,w,h]
        if not(box_w > 0) or not(box_h > 0):
            raise ValueError("Incorrect bbox size: height {}, width {}".format(box_h, box_w))
        delta = [0] * 4
        delta[0] = (box_cx - anchors[aid][0]) / box_w
        delta[1] = (box_cy - anchors[aid][1]) / box_h
        delta[2] = np.log(box_w / anchors[aid][2])
        delta[3] = np.log(box_h / anchors[aid][3])
        delta_per_img.append(delta)
    return delta_per_img


def batch_iou(boxes, box):
    """Compute the Intersection-Over-Union of a batch of boxes with another
    box.

    Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
    Returns:
    ious: array of a float number in range [0, 1].
    """
    lr = np.maximum(np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
                    np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
                    0)

    tb = np.maximum(
        np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
        np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
        0)

    inter = lr*tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter/union



def assert_list(v):
    assert_type(v, list)
    assert len(v)>0, "Error: Provided list is empty."
    return v


def assert_type(v, t):
    assert type(v) == t, "Error: {} provided while {} is expected.".format_map(type(v).__name__, t.__name__)