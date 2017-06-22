#!/usr/bin/env python
"""
SmallNet.py is a class that provides a network for Person localization
Created 6/13/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"



import tensorflow as tf
from tensorflow import name_scope, variable_scope, stop_gradient

from .ObjectDetection import ObjectDetectionNet

class SmallNet(ObjectDetectionNet):

    def __init__(self, labels_provided, imshape, lr=1e-3):

        super().__init__(labels_provided, imshape, lr)

    def _add_featuremap(self):


        conv = self._conv2d
        deconv = self._deconv
        concat = self._concat
        maxpool = self._max_pool
        bn = self._batch_norm
        fc = self._fullyconnected

        def upsampling(input, filters, name):

            with variable_scope('upsampling_' + name):
                input = bn(input, name=name)
                with variable_scope('tower1'):
                    t1 = conv(input, filters, name='conv1')
                    t1 = conv(t1, filters, name='conv2')
                    t1 = bn(t1, name='batch_norm')
                    t1 = conv(t1, filters, name='conv3')
                    t1 = conv(t1, filters, name='conv4')
                    t1 = maxpool(t1, name='maxpool', padding='SAME')

                with variable_scope('tower2'):
                    t2 = conv(input, filters, name='conv1')
                    t2 = conv(t2, filters, name='conv2')
                    t2 = maxpool(t2, name='maxpool', padding='SAME')

                with variable_scope('tower3'):
                    t3 = conv(input, filters, bias=False, name='regularization')
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
            conv1 = conv(inputs, 8, name='conv1')
            conv1 = conv(conv1, 8, 'conv1')
            conv1 = maxpool(conv1, name = 'maxpool', padding='SAME')


        up1 = upsampling(conv1, 8, 'up1')
        up2 = upsampling(up1, 16, 'up2')
        up3 = upsampling(up2, 32, 'up3')
        up4 = upsampling(up3, 64, 'up4')
        up5 = upsampling(up4, 64, 'up5')

        dw4 = downsampling(up5, 64, 'd4')
        dw4 = lateral_connection(dw4, up4, 64, 'tdm4')
        dw3 = downsampling(dw4, 32, 'd3')
        dw3 = lateral_connection(dw3, up3, 64, 'tdm3')
        dw2 = downsampling(dw3, 32, 'd2')
        dw2 = lateral_connection(dw2, up2, 32, 'tdm2')
        dw1 = downsampling(dw2, 32, 'd1')
        dw1 = lateral_connection(dw1, up1, 32, 'tdm1')

        self.featuremap = conv(dw1, self.K*(self.n_classes + 4 + 1), name='featuremap')