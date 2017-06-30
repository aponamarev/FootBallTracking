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
        self.outshape = Point(int(imshape[0] / 32), int(imshape[1] / 32))

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

            tf.summary.image("imgs", inputs, max_outputs=2)

            inputs = tf.subtract( tf.divide(inputs, 255.0), 0.5, name="img_norm")



        with name_scope('inputs'):

            tf.summary.image("imgs", inputs, max_outputs=2)

            inputs = tf.subtract( tf.divide(inputs, 255.0), 0.5, name="img_norm")

        c = conv(inputs, 8, BN_FLAG=False, name='conv1')
        c = separable_conv(c, 32, BN_FLAG=True, strides=2, name='conv2')
        c = separable_conv(c, int(32*self.width), BN_FLAG=True, name='conv3')
        c = separable_conv(c, int(64*self.width), BN_FLAG=True, strides=2, name='conv4')
        c = separable_conv(c, int(64*self.width), BN_FLAG=True, name='conv5')
        c = separable_conv(c, int(128*self.width), BN_FLAG=True, strides=2, name='conv6')
        c = separable_conv(c, int(128*self.width), BN_FLAG=True, name='conv7')
        c = separable_conv(c, int(256*self.width), BN_FLAG=True, strides=2, name='conv8')
        c = separable_conv(c, int(256*self.width), BN_FLAG=True, name='conv9')
        c = separable_conv(c, int(512*self.width), BN_FLAG=True, strides=2, name='conv10')

        self.feature_map = separable_conv(c, self.K * (self.n_classes + 4 + 1), name='feature_map')