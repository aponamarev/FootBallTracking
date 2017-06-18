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

    def __init__(self, labels_provided, batch_sz, imshape):

        super().__init__(labels_provided, batch_sz, imshape)

    def _add_featuremap(self):


        conv = self._conv2d
        deconv = self._deconv
        concat = self._concat
        maxpool = self._max_pool
        bn = self._batch_norm
        fc = self._fullyconnected

        def upsampling(input, filters, name):

            with variable_scope('upsampling/' + name):
                input = bn(input, name=name)
                shape = input.get_shape().as_list()
                t1 = conv(input, [3, 3, shape[3], filters], name='t1/conv1')
                t1 = conv(t1, [3, 3, filters, filters], name='t1/conv2')
                t1 = conv(t1, [3, 3, filters, filters], name='t1/conv3')
                t1 = conv(t1, [3, 3, filters, filters], name='t1/conv4')
                t1 = maxpool(t1, name='t1/maxpool', padding='SAME')

                t2 = conv(input, [3, 3, shape[3], filters], name='t2/conv1')
                t2 = conv(t2, [3, 3, filters, filters], name='t2/conv2')
                t2 = maxpool(t2, name='t2/maxpool', padding='SAME')

                t3 = conv(input, [1, 1, shape[3], filters], name='t3/conv1')
                t3 = maxpool(t3, name='t3/maxpool', padding='SAME')

                c = concat([t1, t2, t3], axis=3)
                c_shape = c.get_shape().as_list()
                c = conv(c, [1,1,c_shape[3],filters], name="output")

            return c

        def downsampling(input, filters, name):
            shape = input.get_shape().as_list()

            with variable_scope('downsampling/' + name):
                input = bn(input, name=name)
                d = deconv(input, filters, [3,3], [2,2], padding='SAME')
                d = conv(d, [3,3,filters,filters], name='output')

            return d

        def lateral_connection(td, dt, filters, name):

            with variable_scope('lateral/'+name):
                dt = stop_gradient(dt, name="stop_G")
                dt_shape = dt.get_shape().as_list()
                l = conv(dt, [3,3, dt_shape[3],filters], name="L")
                output = concat((td, l))
                out_shape = output.get_shape().as_list()
                return conv(output, [3,3,out_shape[3], filters], name="force_choice")




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
        dw5 = lateral_connection(dw5, up5, 128, 'tdm5')
        dw4 = downsampling(up5, 64, 'd4')
        dw4 = lateral_connection(dw4, up4, 64, 'tdm4')
        dw3 = downsampling(up4, 32, 'd3')
        dw3 = lateral_connection(dw3, up3, 32, 'tdm3')
        dw2 = downsampling(up3, 32, 'd2')
        dw2 = lateral_connection(dw2, up2, 32, 'tdm2')

        dw2_shape = dw2.get_shape().as_list()

        self.featuremap = conv(dw2, [3, 3, dw2_shape[3], self.K*(self.n_classes + 4 + 1)])













