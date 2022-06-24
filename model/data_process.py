# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
from model.hyparameter import parameter

class dataIterator():             #切记这里的训练时段和测试时段的所用的对象不变，否则需要重复加载数据
    def __init__(self, hp=None):
        '''
        :param is_training: while is_training is True,the model is training state
        :param field_len:
        :param time_size:
        :param prediction_size:
        :param target_site:
        '''
        self.para=hp
        self.site_id=self.para.target_site_id                   # ozone ID
        self.time_size=self.para.input_length                   # time series length of input
        self.prediction_size=self.para.output_length            # the length of prediction
        self.is_training=self.para.is_training                  # true or false
        self.window_step=self.para.step                         # windows step
        self.train_data= self.sudden_changed(np.array(self.get_source_data(self.para.train_path).values[:,2:],dtype=np.float32))
        self.test_data = self.sudden_changed(self.get_source_data(self.para.test_path).values[:,2:])

        # self.data=self.source_data.loc[self.source_data['ZoneID']==self.site_id]
        # self.data=self.source_data
        self.train_length=self.train_data.shape[0]  # train data length
        self.test_length = self.test_data.shape[0]  # test data length
        self.max,self.min=self.get_max_min(self.train_data,self.test_data)   # max and min are list type, used for the later normalization

        self.normalize=self.para.normalize
        if self.normalize:
            self.train_data=self.normalization(self.train_data) #normalization
            self.test_data = self.normalization(self.test_data)  # normalization

    def sudden_changed(self, data):
        '''
        :param data:
        :return:
        '''
        sundden = [300, 300, 300, 300, 300, 300, 40]
        shape = data.shape
        for j in range(shape[1]):
            for i in range(shape[0]):
                if i != 0:
                    if float(data[i][j])- float(data[i - 1][j]) > sundden[j]:
                        data[i][j] = data[i - 1][j]
        return data

    def get_source_data(self,file_path):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        # print(data.values)
        return data

    def get_max_min(self, train_data=None,test_data=None):
        '''
        :return: the max and min value of input features
        '''
        self.min_list=[]
        self.max_list=[]

        for i in range(train_data.shape[1]):
            self.min_list.append(min([min(train_data[:,i]),min(test_data[:,i])]))
            self.max_list.append(max([max(train_data[:,i]),max(test_data[:,i])]))
        print('the max feature list is :',self.max_list)
        print('the min feature list is :', self.min_list)
        return self.max_list,self.min_list

    def normalization(self, data):
        for i in range(data.shape[1]):
            data[:,i]=(data[:,i] - np.array(self.min[i])) / (np.array(self.max[i]) - np.array(self.min[i]))
        return data

    def generator(self):
        '''
        :return: yield the data of every time,
        shape:input_series:[time_size,field_size]
        label:[predict_size]
        '''
        # print('is_training is : ', self.is_training)
        if self.is_training:
            low,high=0,int(self.train_data.shape[0]//self.para.site_num)*self.para.site_num
            data=self.train_data
        else:
            low,high=0,int(self.test_data.shape[0]//self.para.site_num) * self.para.site_num
            data=self.test_data

        while low+self.para.site_num*(self.para.input_length + self.para.output_length)<= high:
            label=data[low + self.time_size * self.para.site_num: low + self.time_size * self.para.site_num + self.prediction_size * self.para.site_num,1: 2]
            label=np.concatenate([label[i * self.para.site_num:(i + 1) * self.para.site_num, :] for i in range(self.prediction_size)], axis=1)

            time=data[low + self.time_size * self.para.site_num: low + self.time_size * self.para.site_num + self.prediction_size * self.para.site_num,1:4]
            time=np.concatenate([time[i * self.para.site_num:(i + 1) * self.para.site_num, :] for i in range(self.prediction_size)], axis=1)

            x_input=np.array(data[low:low+self.time_size*self.para.site_num])

            # a=np.concatenate([x_input[i*self.para.site_num] for i in range(self.para.input_length)],axis=0)
            #
            # x_input=a

            yield (x_input,
                   label[0])
            if self.is_training:
                low += self.window_step*self.para.site_num
            else:
                low+=self.prediction_size*self.para.site_num
        return

    def next_batch(self,batch_size, epochs, is_training=True):
        '''
        :return the iterator!!!
        :param batch_size:
        :param epochs:
        :return:
        '''
        self.is_training=is_training
        dataset=tf.data.Dataset.from_generator(self.generator,output_types=(tf.float32,tf.float32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=int(self.train_data.shape[0]//self.para.site_num -self.time_size-self.prediction_size)//self.window_step)
            dataset=dataset.repeat(count=epochs)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()


def re_current(line, max, min):
    return [[line_sub[i]*(max[i]-min[i])+min[i]+0.1 for i in range(len(line_sub))] for line_sub in line]
#
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=dataIterator(site_id=0,is_training=True,normalize=True,time_size=8,prediction_size=3,hp=para)

    next=iter.next_batch(32,1,False)
    with tf.Session() as sess:
        for _ in range(3):
            x,y,time=sess.run(next)
            # print(time.shape)
            # print(time)
            rows = np.reshape(time, [-1, 3,3])
            print(rows.shape)
            rows = np.array([re_current(row_data, [30.0, 23.0, 60.0], [1.0, 0.0, 15.0]) for row_data in rows],dtype=int)
            print(rows)