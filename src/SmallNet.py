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

class SmallNet(ObjectDetectionNet):

    def __init__(self, labels_provided, imshape, lr=1e-3, width=1.0):

        self.width=width

        self.imshape = Point(*imshape)
        self.outshape = Point(int(imshape[0] / 4), int(imshape[1] / 4))

        super().__init__(labels_provided, lr)

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
                    t2 = separable_conv(input, filters, name='conv2')
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



        with variable_scope("input_upsampling"):
            c = conv(inputs, 8, BN_FLAG=False, name='conv1', strides=2)
            c = separable_conv(c, 32, name='conv2')


        up1 = upsampling(c, int(64*self.width), 'up1')
        up2 = upsampling(up1, int(128*self.width), 'up2')
        up3 = upsampling(up2, int(256*self.width), 'up3')
        up4 = upsampling(up3, int(256*self.width), 'up4')

        dw3 = downsampling(up4, int(256*self.width), 'd3')
        dw3 = lateral_connection(dw3, up3, int(256*self.width), 'tdm3')
        dw2 = downsampling(dw3, int(128*self.width), 'd2')
        dw2 = lateral_connection(dw2, up2, int(128*self.width), 'tdm2')
        dw1 = downsampling(dw2, int(64*self.width), 'd1')
        dw1 = lateral_connection(dw1, up1, int(64*self.width), 'tdm1')

        self.feature_map = separable_conv(dw1, self.K * (self.n_classes + 4 + 1), name='feature_map')