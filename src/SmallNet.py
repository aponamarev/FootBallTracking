#!/usr/bin/env python
"""
SmallNet.py is a class that provides a network for Person localization
Created 6/13/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"



import tensorflow as tf
from collections import namedtuple
from tensorflow import name_scope, variable_scope, stop_gradient

from .ObjectDetection import ObjectDetectionNet

Point = namedtuple('Point',['x', 'y'])

class Net(ObjectDetectionNet):

    def __init__(self, labels_provided, imshape, lr=1e-3, activations='elu', width=1.0):

        self.width=width

        self.imshape = Point(*imshape)
        self.outshape = Point(int(imshape[0] / 64), int(imshape[1] / 64))

        super().__init__(labels_provided, lr)

        self.default_activation = activations



    def _add_featuremap(self):


        separable_conv = self._separable_conv2d
        conv = self._conv2d
        deconv = self._deconv
        concat = self._concat
        maxpool = self._max_pool
        bn = self._batch_norm
        fc = self._fullyconnected

        def upsampling(input, filters, name):

            with variable_scope('upsampling_' + name):
                with variable_scope('tower1'):
                    t1 = separable_conv(input, filters, name='conv1')
                    t1 = separable_conv(t1, filters, name='conv2')
                    t1 = separable_conv(t1, filters, name='conv3')
                    t1 = separable_conv(t1, filters, strides=2, name='conv4')

                with variable_scope('tower2'):
                    t2 = separable_conv(input, filters, name='conv1')
                    t2 = separable_conv(t2, filters, strides=2, name='conv2')

                with variable_scope('tower3'):
                    t3 = separable_conv(input, filters, strides=2, name='regularization')

                c = concat([t1, t2, t3], axis=3, name='concat')
                c = separable_conv(c, filters, name="output")

            return c

        def downsampling(input, filters, name):

            with variable_scope('downsampling_' + name):
                d = deconv(input, filters, [3,3], [2,2], padding='SAME')
                d = separable_conv(d, filters, name='output')

            return d

        def lateral_connection(td, dt, filters, name):

            with variable_scope('lateral_'+name):
                dt = stop_gradient(dt, name="stop_G")
                l = separable_conv(dt, filters, name="L")
                output = concat((td, l))
                return separable_conv(output, filters, name="force_choice")

        inputs = self.input_img

        with name_scope('inputs'):

            inputs = tf.subtract( tf.divide(inputs, 127.5), 1.0, name="img_norm")

        c1 = conv(inputs, 8, strides=2, BN_FLAG=False, name='conv1')
        c2 = separable_conv(c1, 32, BN_FLAG=True, strides=2, name='conv2')
        c3 = upsampling(c2, 64, name="up3")
        c4 = separable_conv(c3, 128, BN_FLAG=True, strides=2, name='conv4')
        c5 = upsampling(c4, 256, name="up5")
        c6 = separable_conv(c5, 512, BN_FLAG=True, strides=2, name='conv6')

        self.feature_map = separable_conv(c6, self.K * (self.n_classes + 4 + 1), name='feature_map')