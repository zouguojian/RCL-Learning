# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 11:07:40 2018
此模型使用的卷积核大小分别是1,2,3
@author: Administrator
"""
import tensorflow as tf
#tf.reset_default_graph()
class Resnet(object):
    def __init__(self,inputs,batch_size):
        self.inputs=inputs
        self.batch_size=batch_size
#        第一层卷积所需要的一些参数
        self.CONV1=1
        self.NUM_CHANNELS=1
        self.CONV1_DEEP=3
        
        self.CONV2=2
        self.CONV2_DEEP=3
        
        self.CONV3=3
        self.CONV3_DEEP=3
    def CNN_layer(self):
        with tf.variable_scope('layer_one_1',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [self.CONV1,self.CONV1,self.NUM_CHANNELS,self.CONV1_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[self.CONV1_DEEP],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(self.inputs, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer1=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))

        with tf.variable_scope('layer_one_2',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [self.CONV2,self.CONV2,self.CONV1_DEEP,self.CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[self.CONV2_DEEP],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer1, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer2=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_3',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [self.CONV3,self.CONV3,self.CONV2_DEEP,self.CONV3_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[self.CONV3_DEEP],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer2, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer3=tf.nn.bias_add(conv_one, bias_one)
        with tf.variable_scope('layer_add',reuse=tf.AUTO_REUSE):
            weight=tf.get_variable("weight",
                                       [self.CONV1,self.CONV1,self.NUM_CHANNELS,3],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("bias",[3],
                                  initializer=tf.constant_initializer(0))
            conv=tf.nn.conv2d(self.inputs, weight, strides=[1, 1, 1, 1], 
                                      padding='SAME')
#            return tf.nn.relu(tf.nn.bias_add(conv, bias)+layer3)
            layer_add=tf.nn.bias_add(conv, bias)+layer3
#            return layer_add
           
        with tf.variable_scope('layer_one_4',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,3,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer_add, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer4=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_5',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer4, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer5=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_6',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer5, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer6=tf.nn.bias_add(conv_one, bias_one)
        with tf.variable_scope('layer_add1',reuse=tf.AUTO_REUSE):
            weight=tf.get_variable("weight",
                                       [1,1,3,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv=tf.nn.conv2d(layer_add, weight, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer_add=tf.nn.relu(tf.nn.bias_add(conv, bias)+layer6)
#            return layer_add

          
        with tf.variable_scope('layer_one_7',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer_add, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer7=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_8',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer7, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer8=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_9',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer8, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer9=tf.nn.bias_add(conv_one, bias_one)
        with tf.variable_scope('layer_add2',reuse=tf.AUTO_REUSE):
            weight=tf.get_variable("weight",
                                       [1,1,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv=tf.nn.conv2d(layer_add, weight, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer_add=tf.nn.relu(tf.nn.bias_add(conv, bias)+layer9)
            # return layer_add


        with tf.variable_scope('layer_one_10',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer_add, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer10=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_11',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer10, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer11=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_12',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer11, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer12=tf.nn.bias_add(conv_one, bias_one)
        with tf.variable_scope('layer_add3',reuse=tf.AUTO_REUSE):
            weight=tf.get_variable("weight",
                                       [1,1,6,6],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("bias",[6],
                                  initializer=tf.constant_initializer(0))
            conv=tf.nn.conv2d(layer_add, weight, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer_add=tf.nn.relu(tf.nn.bias_add(conv, bias)+layer12)
            return layer_add