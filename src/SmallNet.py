#!/usr/bin/env python
"""
SmallNet.py is a class that provides a network for Person localization
Created 6/13/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"



import tensorflow as tf
from tensorflow import VariableScope, name_scope

from .ObjectDetection import ObjectDetectionNet

class SmallNet(ObjectDetectionNet):

    def __init__(self, n_classes, batch_sz, imshape, training_mode_flag, dropout_keep_rate):

        super().__init__(n_classes, batch_sz, imshape, training_mode_flag, dropout_keep_rate)

    def _add_featuremap(self):


        conv = self._conv2d
        deconv = self._deconv
        concat = self._concat
        maxpool = self._max_pool
        bn = self._batch_norm
        fc = self._fullyconnected

        def upsampling(input, filters, name):

            with name_scope('upsampling/' + name):
                t1 = conv(inputs, [1, 3, 3, filters])
                t1 = conv(t1, [1, 3, 3, filters])
                t1 = conv(t1, [1, 3, 3, filters])
                t1 = conv(t1, [1, 3, 3, filters])
                t1 = maxpool(t1)

                t2 = conv(inputs, [1, 3, 3, filters])
                t2 = conv(t2, [1, 3, 3, filters])
                t2 = maxpool(t2)

                t3 = maxpool(inputs)

                c = concat((t1, t2, t3))
                c = conv(c, [1,1,1,filters])

            return c

        def downsampling(input, filters, name):

            with name_scope('downsampling/' + name):
                d = deconv(input, [1,3,3,filters], padding='VALID')
                d = conv(d, [1,3,3,filters])

            return d

        inputs = self.input_img

        with name_scope('inputs'):

            tf.summary.image("imgs", inputs, max_outputs=6)

            inputs = tf.subtract( tf.divide(inputs, 255.0), 0.5, name="img_norm")


        up1 = upsampling(inputs, 8, 'up1')
        up2 = upsampling(up1, 16, 'up2')
        up3 = upsampling(up2, 32, 'up3')
        up4 = upsampling(up3, 64, 'up4')
        up5 = upsampling(up4, 128, 'up5')
        up6 = upsampling(up5, 128, 'up6')

        dw5 = downsampling(up6, 128, 'd5')
        dw5 = concat((dw5, up5))
        dw4 = downsampling(up6, 64, 'd4')
        dw4 = concat((dw4, up4))
        dw3 = downsampling(up6, 32, 'd3')
        dw3 = concat((dw3, up3))
        dw2 = downsampling(up6, 16, 'd2')
        dw2 = concat((dw2, up2))
        dw1 = downsampling(up6, 8, 'd1')
        dw1 = concat((dw1, up1))

        self.featuremap = conv(dw1, [1,3,3, self.K + 4 + 1 + self.n_classes])













