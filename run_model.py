# -- coding: utf-8 --

from model.res_net import resnet
from model.multi_convlstm import *
from model.hyparameter import parameter
from model.data_process import dataIterator

import tensorflow as tf
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logs_path="board"
para=parameter(argparse.ArgumentParser()).get_para()

class Model(object):
    def __init__(self,para):
        self.para=para

        self.iterate = dataIterator(site_id=self.para.target_site_id,
                                                is_training=True,
                                                time_size=self.para.input_length,
                                                prediction_size=self.para.output_length,
                                                window_step=self.para.step,
                                                normalize=self.para.normalize,
                                                hp=self.para)

        # define placeholders
        self.placeholders = {
            'features': tf.placeholder(tf.float32, shape=[self.para.batch_size*self.para.input_length, self.para.height, self.para.width]),
            'labels': tf.placeholder(tf.float32, shape=[self.para.batch_size, self.para.output_length]),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'is_training':tf.placeholder_with_default(input=False,shape=())
        }
        self.model()

    def model(self):
        '''
        :param batch_size: 64
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: True
        :return:
        '''

        # create model

        l = resnet(batch_size=self.para.batch_size,para=para)
        x_input = self.placeholders['features']
        inputs = tf.reshape(x_input, shape=[-1, self.para.input_length, self.para.height, self.para.width])
        '''
        cnn output shape is : [batch size, height, site num, output channel]
        '''
        cnn_out = tf.concat([tf.expand_dims(l.cnn(tf.expand_dims(inputs[:, i, :, :], axis=3)), axis=1) for i in
                             range(self.para.input_length)], axis=1)

        print('resnet output shape is : ',cnn_out.shape)
        '''
        resnet output shape is :  (32, 3, 14, 4, 32)
        '''
        mul_convl=mul_convlstm(batch=self.para.batch_size,
                               predict_time=self.para.output_length,
                               shape=[cnn_out.shape[2],cnn_out.shape[3]],
                               filters=32,
                               kernel=[self.para.height, 2],
                               layer_num=self.para.hidden_layer,
                               normalize=self.para.is_training)

        h_states=mul_convl.encoding(cnn_out)
        self.pres=mul_convl.decoding(h_states)

        self.cross_entropy = tf.reduce_mean(
            tf.sqrt(tf.reduce_mean(tf.square(self.placeholders['labels'] - self.pres), axis=0)))

        print('self.pres shape is : ', self.pres.shape)
        print('labels shape is : ', self.placeholders['labels'].shape)

        print(self.cross_entropy)
        print('cross shape is : ',self.cross_entropy.shape)

        # tf.summary.scalar('cross_entropy',self.cross_entropy)
        # backprocess and update the parameters
        self.train_op = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.cross_entropy)

    def test(self):
        '''
        :param batch_size: usually use 1
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: False
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)


    def accuracy(self,label,predict):
        '''
        :param Label: represents the observed value
        :param Predict: represents the predicted value
        :param epoch:
        :param steps:
        :return:
        '''
        error = label - predict
        average_error = np.mean(np.fabs(error.astype(float)))
        print("mae is : %.6f" % (average_error))

        rmse_error = np.sqrt(np.mean(np.square(label - predict)))
        print("rmse is : %.6f" % (rmse_error))

        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (predict - np.mean(predict)))) / (np.std(predict) * np.std(label))
        print('correlation coefficient is: %.6f' % (cor))

        # mask = label != 0
        # mape =np.mean(np.fabs((label[mask] - predict[mask]) / label[mask]))*100.0
        # mape=np.mean(np.fabs((label - predict) / label)) * 100.0
        # print('mape is: %.6f %' % (mape))
        sse = np.sum((label - predict) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        R2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
        print('r^2 is: %.6f' % (R2))

        return average_error,rmse_error,cor,R2

    def describe(self,label,predict,prediction_size):
        '''
        :param label:
        :param predict:
        :param prediction_size:
        :return:
        '''
        plt.figure()
        # Label is observed value,Blue
        plt.plot(label[0:prediction_size], 'b*:', label=u'actual value')
        # Predict is predicted value，Red
        plt.plot(predict[0:prediction_size], 'r*:', label=u'predicted value')
        # use the legend
        # plt.legend()
        plt.xlabel("time(hours)", fontsize=17)
        plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
        plt.title("the prediction of pm$_{2.5}", fontsize=17)
        plt.show()

    def initialize_session(self):
        self.sess=tf.Session()
        self.saver=tf.train.Saver()

    def re_current(self, a, max, min):
        return [num*(max-min)+min for num in a]

    def construct_feed_dict(self, features, labels, placeholders):
        """Construct feed dictionary."""
        feed_dict = dict()
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['features']: features})
        return feed_dict

    def run_epoch(self):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''

        max_rmse = 100
        self.sess.run(tf.global_variables_initializer())
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
        #
        # for (x, y) in zip(tf.global_variables(), self.sess.run(tf.global_variables())):
        #     print('\n', x, y)

        iterate=self.iterate
        next_elements=iterate.next_batch(batch_size=self.para.batch_size,epochs=self.para.epochs,is_training=True)

        # '''
        for i in range(int((iterate.train_length //self.para.site_num-(iterate.time_size + iterate.prediction_size))//iterate.window_step)
                       * self.para.epochs // self.para.batch_size):
            x, label =self.sess.run(next_elements)

            # Construct feed dictionary
            # features = sp.csr_matrix(x)
            # features = preprocess_features(features)
            features=np.reshape(np.array(x), [-1, self.para.height, self.para.width])

            feed_dict = self.construct_feed_dict(features, label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.para.dropout})

            loss, _ = self.sess.run((self.cross_entropy,self.train_op), feed_dict=feed_dict)
            # writer.add_summary(summary, loss)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss))

            # '''
            # validate processing
            if i % 10 == 0 and i>0:
                rmse_error=self.evaluate()

                if max_rmse>rmse_error:
                    print("the validate average rmse loss value is : %.6f" % (rmse_error))
                    max_rmse=rmse_error
                    self.saver.save(self.sess,save_path=self.para.save_path+'model.ckpt')

    def evaluate(self):
        '''
        :param para:
        :param pre_model:
        :return:
        '''
        label_list = list()
        predict_list = list()

        model_file = tf.train.latest_checkpoint(self.para.save_path)
        if not self.para.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)

        iterate_test =self.iterate
        next_ = iterate_test.next_batch(batch_size=self.para.batch_size, epochs=1,is_training=False)
        max,min=iterate_test.max_list[1],iterate_test.min_list[1]


        # '''
        for i in range(int((iterate_test.test_length // self.para.site_num
                            -(iterate_test.time_size + iterate_test.prediction_size))//iterate_test.prediction_size)// self.para.batch_size):
            x, label =self.sess.run(next_)

            features=np.reshape(np.array(x), [-1, self.para.height, self.para.width])
            feed_dict = self.construct_feed_dict(features, label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})    #不能取 1.0，因为我们使用的是1-dropout为正则的方式，可取 0.0
            # feed_dict.update({self.placeholders['is_training']:self.para.is_training})

            pre = self.sess.run((self.pres), feed_dict=feed_dict)
            label_list.append(label)
            predict_list.append(pre)

        label_list=np.reshape(np.array(label_list,dtype=np.float32),[-1, self.para.output_length])
        predict_list=np.reshape(np.array(predict_list,dtype=np.float32),[-1, self.para.output_length])


        if self.para.normalize:
            label_list = np.array([self.re_current(np.reshape(site_label, [-1]),max,min) for site_label in label_list],dtype=np.float32)
            predict_list = np.array([self.re_current(np.reshape(site_label, [-1]),max,min) for site_label in predict_list],dtype=np.float32)
        else:
            label_list = np.array([np.reshape(site_label, [-1]) for site_label in label_list],dtype=np.float32)
            predict_list = np.array([np.reshape(site_label, [-1]) for site_label in predict_list],dtype=np.float32)

        label_list=np.reshape(label_list,[-1])
        predict_list=np.reshape(predict_list,[-1])
        average_error, rmse_error, cor, R2 = self.accuracy(label_list, predict_list)  #产生预测指标
        #pre_model.describe(label_list, predict_list, pre_model.para.prediction_size)   #预测值可视化
        return rmse_error

def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    print('beginning____________________________beginning_____________________________beginning!!!')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:para.is_training = True
    else:
        para.batch_size=1
        para.is_training = False

    pre_model = Model(para)
    pre_model.initialize_session()

    if int(val) == 1:pre_model.run_epoch()
    else:
        pre_model.evaluate()

    print('finished____________________________finished_____________________________finished!!!')

def re_current(a, max, min):
    print(a.shape)
    return [float(num*(max-min)+min) for num in a]

if __name__ == '__main__':
    main()