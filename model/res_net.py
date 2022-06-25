# -- coding: utf-8 --
import tensorflow as tf


class resnet(object):
    def __init__(self, para=None):
        '''
        :param batch_size:
        :param para:
        '''
        self.para = para
        self.h = [3, 3, 3]
        self.w = [3, 3, 3]
        self.out_channel = [3, 3, 3]
        self.in_channel = [1] + self.out_channel
        self.features = 5

    def conv_2d(self, x, h, w, in_channel, out_channel, layer_name):
        '''
        :param x:
        :param h:
        :param w:
        :param in_channel:
        :param out_channel:
        :return:
        '''
        filter = tf.Variable(initial_value=tf.truncated_normal(shape=[h, w, in_channel, out_channel], stddev=0.1),
                             name=layer_name)
        # filter = tf.get_variable(name=layer_name, shape=[h, w, in_channel, out_channel],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable(layer_name + "/bias", [out_channel], initializer=tf.constant_initializer(0))
        layer = tf.nn.conv2d(input=x, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.bias_add(layer, bias)

    def relu(self, inputs):
        '''
        :param inputs:
        :return:
        '''
        relu = tf.nn.relu(inputs)
        return relu

    def block(self, x, in_channel, out_channel, block_name):
        '''
        :param x:
        :param in_channel:
        :param out_channel:
        :param block_name:
        :return:
        '''
        x1 = self.conv_2d(x, self.h[0], self.w[0], in_channel[0], out_channel[0], block_name + '/1')
        x1 = self.relu(x1)
        x2 = self.conv_2d(x1, self.h[0], self.w[0], in_channel[1], out_channel[1], block_name + '/2')
        x2 = self.relu(x2)
        x3 = self.conv_2d(x2, self.h[0], self.w[0], in_channel[2], out_channel[2], block_name + '/3')
        x3 = self.relu(x3)
        return x3

    def residual_connected(self, x1, x2, h, w, in_channel, out_channel, residual_name):
        '''
        :param x1:
        :param x2:
        :param h:
        :param w:
        :param in_channel:
        :param out_channel:
        :param residual_name:
        :return:
        '''
        filter = tf.Variable(initial_value=tf.random_normal(shape=[h, w, in_channel, out_channel]), name=residual_name)
        bias = tf.get_variable(residual_name + "/bias", [out_channel], initializer=tf.constant_initializer(0))
        conv = tf.nn.conv2d(x1, filter, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, bias)
        layer_add = conv + x2
        return self.relu(layer_add)

    def cnn(self, x):
        '''
        :param x: [batch size, site num, features, channel]
        :return: [batch size, height, channel]
        '''

        with tf.variable_scope(name_or_scope='resnet', reuse=tf.AUTO_REUSE):
            block1 = self.block(x, [1, 3, 3], [3, 3, 3], block_name='block1')
            residual1 = self.residual_connected(x, block1, 1, 1, 1, 3, residual_name='residual1')
            print('residual 1 shape is : ', residual1.shape)

            block2 = self.block(residual1, [3, 6, 6], [6, 6, 6], block_name='block2')
            residual2 = self.residual_connected(residual1, block2, 1, 1, 3, 6, residual_name='residual2')
            print('residual 2 shape is : ', residual2.shape)

            block3 = self.block(residual2, [6, 6, 6], [6, 6, 6], block_name='block3')
            residual3 = self.residual_connected(residual2, block3, 1, 1, 6, 6, residual_name='residual3')
            print('residual 3 shape is : ', residual3.shape)

            block4 = self.block(residual3, [6, 6, 6], [6, 6, 6], block_name='block4')
            residual4 = self.residual_connected(residual3, block4, 1, 1, 6, 6, residual_name='residual4')
            print('residual 4 shape is : ', residual4.shape)

            # max_pool=tf.nn.avg_pool(residual4, ksize=[1, 2, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

            max_pool = residual4
            print('max_pool output shape is : ', max_pool.shape)

        # cnn_shape = max_pool3.get_shape().as_list()
        # nodes = cnn_shape[1] * cnn_shape[2] * cnn_shape[3]
        # reshaped = tf.reshape(max_pool3, [cnn_shape[0], nodes])

        return max_pool


if __name__ == '__main__':
    batch_size = 32
    timesteps = 3
    shape = [162, 5]
    kernel = [162, 2]
    channels = 1
    filters = 12  # numbers of output channel

    # Create a placeholder for videos.
    inputs = tf.placeholder(tf.float32, [batch_size, 1, 7, 1])

    multi = resnet(batch_size=32, para=None)
    multi.cnn(inputs)