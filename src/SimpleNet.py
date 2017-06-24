#!/usr/bin/env python
"""
Simple.py is a class that provides a network for Person localization
Created 6/13/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"



import tensorflow as tf
from collections import namedtuple
from tensorflow import name_scope, variable_scope, stop_gradient

from .ObjectDetection import ObjectDetectionNet

Point = namedtuple('Point',['x', 'y'])

class SimpleNet(ObjectDetectionNet):

    def __init__(self, labels_provided, imshape, lr=1e-3):

        self.imshape = Point(*imshape)
        self.outshape = Point(int(imshape[0] / 8), int(imshape[1] / 8))

        super().__init__(labels_provided, lr)

    def _add_featuremap(self):


        conv = self._conv2d
        deconv = self._deconv
        concat = self._concat
        maxpool = self._max_pool
        bn = self._batch_norm
        fc = self._fullyconnected

        def conv_block(input, filters, name, upsampling=True):

            with variable_scope('upsampling_' + name):
                input = bn(input, name=name)
                with variable_scope('tower1'):
                    t1 = conv(input, max(16, int(filters/4)), name='conv1')
                    t1 = conv(t1, filters, name='conv2')
                    t1 = bn(t1, name='batch_norm')
                    t1 = conv(t1, filters, name='conv3')
                    t1 = conv(t1, filters, name='conv4')
                    if upsampling:
                        t1 = maxpool(t1, name='maxpool', padding='SAME')

                with variable_scope('tower2'):
                    t2 = conv(input, max(16, int(filters/4)), name='conv1')
                    t2 = conv(t2, filters, name='conv2')
                    if upsampling:
                        t2 = maxpool(t2, name='maxpool', padding='SAME')

                with variable_scope('tower3'):
                    t3 = conv(input, filters, bias=False, name='regularization')
                    if upsampling:
                        t3 = maxpool(t3, name='maxpool', padding='SAME')

                c = concat([t1, t2, t3], axis=3, name='concat')
                c = conv(c, filters, name="output")

            return c

        def downsampling(input, filters, name):

            with variable_scope('downsampling_' + name):
                input = bn(input, name='batch_norm')
                d = deconv(input, filters, [3,3], [2,2], padding='SAME')
                d = conv(d, filters, name='output')

            return d

        def lateral_connection(td, dt, filters, name):

            with variable_scope('lateral_'+name):
                dt = stop_gradient(dt, name="stop_G")
                l = conv(dt, filters, name="L")
                output = concat((td, l))
                return conv(output, filters, name="force_choice")




        inputs = self.input_img

        with name_scope('inputs'):

            tf.summary.image("imgs", inputs, max_outputs=2)

            inputs = tf.subtract( tf.divide(inputs, 255.0), 0.5, name="img_norm")



        with variable_scope("input_upsampling"):
            c = conv(inputs, 8, name='conv1')
            c = conv(c, 32, name='conv2', strides=(2,2), padding='SAME')
            c = maxpool(c, name = 'maxpool', padding='SAME')


        up1 = conv_block(c, 32, 'up1')
        up2 = conv_block(up1, 64, 'up2', upsampling=False)
        up3 = conv_block(up2, 64, 'up3')
        up4 = conv_block(up3, 128, 'up4')
        up5 = conv_block(up4, 256, 'up5', upsampling=False)

        self.featuremap = conv(up5, self.K*(self.n_classes + 4 + 1), name='feature_map')