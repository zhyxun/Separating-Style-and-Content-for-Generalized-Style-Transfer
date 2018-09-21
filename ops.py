__author__ = 'yxzhang'
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

def conv(batch_input, out_channels, stride, kernel_size):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        # padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="SAME")
        return conv

def full_connect(batch_input, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = batch_input.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(batch_input, matrix) + bias, matrix, bias
        else:
            return tf.matmul(batch_input, matrix) + bias

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2
        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def relu(x):
    with tf.name_scope("relu"):
        x = tf.identity(x)
        return 0.5 * x + 0.5 * tf.abs(x)

def batchnorm(input, mode):
    with tf.variable_scope("batchnorm"):
        input = tf.identity(input)

        # channels = input.get_shape()[3]
        # offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        # scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        # mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        # variance_epsilon = 1e-5
        # normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)# train_result1
        if mode == 'train':
            normalized = tf.layers.batch_normalization(input,training=True)#train_result2
        else:
            normalized = tf.layers.batch_normalization(input,training=False)#train_result2
        # normalized = tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=True)
        return normalized

def deconv(batch_input, out_channels, kernel_size, stride, add):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        output_size = in_height*stride+add
        filter = tf.get_variable("filter", [kernel_size, kernel_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, output_size, output_size, out_channels], [1, stride, stride, 1], padding="SAME")
        return conv
