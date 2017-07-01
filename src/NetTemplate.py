	# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, batch_norm
from tensorflow.contrib.layers import separable_conv2d, conv2d
from tensorflow import concat
import numpy as np

class NetTemplate(object):
    def __init__(self, default_activation='elu',
                 dtype=tf.float32):
        self.weights = []
        self.size = []
        self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        tf.add_to_collection(tf.GraphKeys.RESOURCES, self.dropout_rate)
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name="is_training")
        tf.add_to_collection(tf.GraphKeys.RESOURCES, self.is_training)
        self.default_activation = default_activation
        self._default_activation_summary = 'img'
        self._dtype = dtype
        self.feature_map=None
        self.total_loss=None
        self.optimization_op=None
        self.activations=[]
        self.debug = True


    def fit(self, feed_dict):
        raise NotImplementedError("Fit method needs to be defined in the child class")

    def eval(self, feed_dict):
        raise NotImplementedError("Eval method needs to be defined in the child class")

    def infer(self, X_batch):
        raise NotImplementedError("Infer method needs to be defined in the child class")


    def _define_prediction(self):
        self.predict_class_op = None
        raise  NotImplementedError("Prediction method needs to be defined in the child class")

    def _define_accuracy(self):
        self.accuracy_op = None
        raise  NotImplementedError("Accuracy method needs to be defined in the child class")


    def get_size(self):
        """
        Estimates the size of weights of the model.
        
        :return: dictionary with 'parameters' and 'Mb'
        """
        params = np.sum(np.sum(self.size))
        return {'parameters':params, 'Mb': params*self._dtype.size/(2.0**20)}

    def _define_net(self):

        raise NotImplementedError("Feature map encoder was not implemented!")

    def _define_loss(self):
        raise NotImplementedError("Loss estimate was not defined!")


    def _conv2d(self, inputs, filters, kernel_size=3, strides=1,
                padding="SAME",name="conv2d", bias=True, BN_FLAG=True):
        """
        kernel_size: An integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window.
        Can be a single integer to specify the same value for all spatial dimensions.
        """
        with tf.variable_scope(name):

            conv = conv2d(inputs=inputs, num_outputs=filters, activation_fn=self._activation(),
                          kernel_size=kernel_size, stride=strides,padding=padding, scope=name)
            self._assert_valid(conv)
            if BN_FLAG:
                conv = self._batch_norm(conv, name='conv_bn')
                self._assert_valid(conv)

        return conv

    def _separable_conv2d(self, inputs, filters, kernel_size=3, strides=1,
                          padding="SAME", name="conv2d", BN_FLAG=True):
        """
        kernel_size: An integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window.
        Can be a single integer to specify the same value for all spatial dimensions.
        """
        with tf.variable_scope(name):

            conv = separable_conv2d(inputs=inputs,
                                    num_outputs=None,
                                    stride=strides,
                                    activation_fn=self._activation(),
                                    kernel_size=[kernel_size, kernel_size],
                                    depth_multiplier=1,
                                    padding=padding,
                                    scope='depthwise_conv')

            self._assert_valid(conv)

            if BN_FLAG:
                conv = self._batch_norm(conv, name='depthwise_bn')
                self._assert_valid(conv)

            conv = conv2d(inputs=conv,
                          num_outputs=filters,
                          activation_fn=self._activation(),
                          kernel_size=[1,1],
                          scope='pointwise_conv')

            self._assert_valid(conv)

            if BN_FLAG:
                conv = self._batch_norm(conv, name='pointwise_bn')
                self._assert_valid(conv)

        return conv


    def _deconv(self, input, filters, kernel_size, strides=[2,2], padding="SAME", name="deconv", bias=True, BN_FLAG=True):

        with tf.variable_scope(name):
            conv = tf.layers.conv2d_transpose(input, filters, kernel_size, strides, padding, activation=self._activation(),
                                              use_bias=bias, kernel_initializer=xavier_initializer(), name="deconv")
            self._assert_valid(conv)

            if BN_FLAG:
                conv = self._batch_norm(conv, name='deconv_bn')
                self._assert_valid(conv)

        return conv


    def _concat(self, *arg, axis=3, name='merge'):

        return concat(*arg, axis=axis, name=name)


    def _fullyconnected(self, inputs, output_channels, name="fully_connected", bias=True, dtype=None):
        dtype = dtype or self._dtype

        X = tf.contrib.layers.flatten(inputs)
        shapes = X.get_shape().as_list()
        shapes = [shapes[1], output_channels]

        # Create weights and biases
        with tf.variable_scope(name):
            W = tf.get_variable("W",
                                shapes,
                                dtype=dtype,
                                initializer=tf.contrib.layers.xavier_initializer())

            self._collect_parameter_stats(W)

            line = tf.matmul(X, W)

            if bias:
                b_size = shapes[1]
                b_init = tf.zeros(b_size, dtype=dtype)
                b = tf.Variable(b_init, name="b")

                self._collect_parameter_stats(b)

                line = tf.nn.bias_add(line, b, data_format='NHWC')



            return self._activation(line)

    def _max_pool(self, inputs, kernel=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name = "max_pool"):
        return tf.nn.max_pool(inputs, kernel, strides, padding=padding, name=name)

    def _avg_pool(self, inputs, kernel=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name = "max_pool"):
        return tf.nn.avg_pool(inputs, kernel, strides, padding=padding, name=name)

    def _batch_norm(self, input, name=None, trainable=True):
        return tf.cond(tf.equal(self.is_training, tf.constant(True)),
                       lambda: batch_norm(input, trainable=trainable, is_training=True, updates_collections=None, scope=name),
                       lambda: batch_norm(input, trainable=trainable, is_training=False, reuse=True, updates_collections=None, scope=name))

    def _drop_out_fullyconnected(self, input, name):
        return tf.nn.dropout(input, self.dropout_rate, name=name)

    def _drop_out_conv(self, input, name):
        shape = input.get_shape().as_list()
        shape[0] = -1
        with tf.name_scope(name):
            flat = tf.contrib.layers.flatten(input)
            dropout = self._drop_out_fullyconnected(flat, name="dropout")
            conv = tf.reshape(dropout, shape=shape)
        return conv

    def _relu_activation(self):

        activation = tf.nn.relu

        return activation

    def _elu_activation(self):

        activation = tf.nn.elu

        return activation

    def _activation_summary(self, activation):
        type = self._default_activation_summary

        implemented_types = {
            'img': tf.summary.image,
            'hist': tf.summary.histogram
        }

        assert type in implemented_types.keys(), "Incorrect type provided ({}). Only {} types are implemented at the moment". \
            format(type, implemented_types.keys())

        return implemented_types[type]('{}_img'.format(activation.op.name), activation)

    def _activation(self, type=None):
        type = type or self.default_activation
        implemented_types = {
            'elu': tf.nn.elu,
            'relu': tf.nn.relu
        }
        assert type in implemented_types.keys(), "Incorrect type provided ({}). Only {} types are implemented at the moment".\
            format(type, implemented_types.keys())

        activation = implemented_types[type]

        return activation

    def _collect_parameter_stats(self, variable):
        shape = variable.get_shape().as_list()
        size = np.product(shape, dtype=np.int)
        tf.summary.histogram('{}_hist'.format(variable.op.name), variable)
        self.weights.append(variable)
        self.size.append(size)

    def _assert_valid(self, x):
        if self.debug:
            check = tf.verify_tensor_all_finite(x, "incorrect value provided in {}".format(x.op.name), name=x.op.name + "/assert_nan_or_inf")
            tf.add_to_collection("Assertion", check)

