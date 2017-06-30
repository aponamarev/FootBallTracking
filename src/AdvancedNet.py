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

class AdvancedNet(ObjectDetectionNet):

    def __init__(self, labels_provided, imshape, lr=1e-3, activations='elu', width=1.0):

        self.width=width

        self.imshape = Point(*imshape)
        self.outshape = Point(int(imshape[0] / 8), int(imshape[1] / 8))

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
                input = separable_conv(input, filters, strides=2, name='upsampling')
                with variable_scope('tower1'):
                    t1 = separable_conv(input, int(filters*self.width), name='conv1')
                    t1 = separable_conv(t1, int(filters*self.width), name='conv2')
                    t1 = separable_conv(t1, int(filters*self.width), name='conv3')
                    t1 = separable_conv(t1, int(filters*self.width), name='conv4')

                with variable_scope('tower2'):
                    t2 = separable_conv(input, int(filters*self.width), name='conv1')
                    t2 = separable_conv(t2, int(filters*self.width), name='conv2')

                with variable_scope('tower3'):
                    t3 = separable_conv(input, int(filters*self.width), name='regularization')

                c = concat([t1, t2, t3], axis=3, name='concat')
                c = separable_conv(c, int(filters*self.width), name="output")

            return c

        def downsampling(input, filters, name):

            with variable_scope('downsampling_' + name):
                d = deconv(input, int(filters*self.width), [3,3], [2,2], padding='SAME')
                d = separable_conv(d, int(filters*self.width), name='output')

            return d

        def lateral_connection(td, dt, filters, name):

            with variable_scope('lateral_'+name):
                dt = stop_gradient(dt, name="stop_G")
                l = separable_conv(dt, int(filters*self.width), name="L")
                output = concat((td, l))
                return separable_conv(output, int(filters*self.width), name="force_choice")

        inputs = self.input_img

        with name_scope('inputs'):

            tf.summary.image("imgs", inputs, max_outputs=2)

            inputs = tf.subtract( tf.divide(inputs, 255.0), 0.5, name="img_norm")



        with name_scope('inputs'):

            tf.summary.image("imgs", inputs, max_outputs=2)

            inputs = tf.subtract( tf.divide(inputs, 255.0), 0.5, name="img_norm")

        c1 = conv(inputs, 8, BN_FLAG=False, name='conv1')
        c2 = separable_conv(c1, 32, BN_FLAG=True, strides=2, name='conv2')
        c3 = upsampling(c2, 64, name="upsampling3")
        c4 = separable_conv(c3, 128, BN_FLAG=True, strides=2, name='conv4')
        c5 = upsampling(c4, 256, name="upsampling5")
        c6 = separable_conv(c5, 512, BN_FLAG=True, strides=2, name='conv6')
        d5 = downsampling(c6, 256, name="downsampling5")
        d5 = lateral_connection(d5, c5, 128, name="l5")
        d4 = downsampling(d5, 128, name="downsampling4")
        d4 = lateral_connection(d4, c4, 64, name="l4")

        self.feature_map = separable_conv(d4, self.K * (self.n_classes + 4 + 1), name='feature_map')