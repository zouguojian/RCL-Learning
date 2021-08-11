# -- coding: utf-8 --
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
file='/Users/guojianzou/Documents/program/shanghai_weather/'


def correlation(obseved_v,predicted_v):

    cor = np.mean(np.multiply((obseved_v - np.mean(obseved_v)),
                              (predicted_v - np.mean(predicted_v)))) / (
                  np.std(predicted_v) * np.std(obseved_v))
    # print(obseved_v.shape,predicted_v.shape)
    # cor=np.corrcoef(obseved_v,predicted_v)
    print('the correlation is : ',cor)

cities=['NanJing','SuZhou','NanTong','WuXi','ChangZhou','ZhenJiang',
        'HangZhou','NingBo','ShaoXing','HuZhou','JiaXing','TaiZhou','ZhouShan']

pollution=pd.read_csv(file+'train_around_weather.csv')

for i in range(3,9):
    for city in cities:
        data1=pollution.loc[pollution['location']=='ShangHai'].values[:,i]
        data2=pollution.loc[pollution['location']==city].values[:,i]
        correlation(data1,data2)

    print('finish')


pollution=pd.read_csv(file+'train_weather_day.csv').values[:,2:]

# print(pollution.values.shape)
#
# data=pollution.loc[pollution['PM2.5']>75]
#
# print(data.values.shape)
#
# print(1088.0/7936.0)

sundden=[300,300,300,300,300,300,40]

def sudden_changed(data):
    '''
    用于处理突变的值
    Args:
        city_dictionary:
    Returns:
'''
    shape=data.shape
    print(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i!=0:
                if data[i][j]-data[i-1][j]>sundden[j]:
                    data[i][j] = data[i - 1][j]
    return data

pollution=sudden_changed(pollution)


weather=pd.read_csv(file+'weather.csv').values[:,1:]

weathers=[]

for line in weather:
    for i in range(3):
        weathers.append(line)

weather=np.array(weathers)[:7937, :]

# plt.figure()
# # Label is observed value,Blue
# font = {'family': 'Times New Roman',
#         'weight': 'normal',
#         'size': 10,
#         }
# font1 = {'family': 'Times New Roman',
#         'weight': 'normal',
#         'size': 8,
#         }
#
# plt.subplot(2, 4, 1)
# plt.plot(pollution[:,0],color='orange', label=u'AQI')
# # use the legend
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# # plt.xlabel("time(hours)", fontsize=17,fontdict=font)
# plt.ylabel("AQI", fontdict=font)
# # plt.title('sample 1 (48 - 40 h)', fontdict=font)
# # plt.title("the prediction of PM2.5", fontsize=17,fontdict=font)
#
# plt.subplot(2, 4, 2)
# plt.plot(pollution[:,1], color='#0cdc73', label=u'PM$_{2.5}$')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 2 (48 - 40 h)', fontdict=font)
#
# plt.subplot(2, 4, 3)
# plt.plot(pollution[:,2], 'b', label=u'PM$_{10}$')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{10}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 3 (48 - 40 h)', fontdict=font)
#
# plt.subplot(2, 4, 4)
# plt.plot(pollution[:,3], color='#f504c9', label=u'SO$_2$')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("SO$_2$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(2, 4, 5)
# plt.plot(pollution[:,4], color='#d0c101', label=u'NO$_2$')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("NO$_2$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(2, 4, 6)
# plt.plot(pollution[:,5],  color='#ff5b00', label=u'O$_3$')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("O$_3$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(2, 4, 7)
# plt.plot(pollution[:,6], color='#a8a495', label=u'CO')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("CO(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.show()



#
# plt.figure()
# # Label is observed value,Blue
# font = {'family': 'Times New Roman',
#         'weight': 'normal',
#         'size': 10,
#         }
# font1 = {'family': 'Times New Roman',
#         'weight': 'normal',
#         'size': 8,
#         }
#
# plt.subplot(3, 4, 1)
# plt.plot(weather[:,0],color='#7FFFD4', label=u'Temperature')
# # use the legend
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# # plt.xlabel("time(hours)", fontsize=17,fontdict=font)
# plt.ylabel("Temperature($°$C)", fontdict=font)
# # plt.title('sample 1 (48 - 40 h)', fontdict=font)
# # plt.title("the prediction of PM2.5", fontsize=17,fontdict=font)
#
# plt.subplot(3, 4, 2)
# plt.plot(weather[:,1], color='#DEB887', label=u'Humidity')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("Humidity(%)", fontdict=font)
# # plt.title('sample 2 (48 - 40 h)', fontdict=font)
#
# plt.subplot(3, 4, 3)
# plt.plot(weather[:,2], color='#7FFF00', label=u'Air pressure')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("Air pressure(Hpa)", fontdict=font)
# # plt.title('sample 3 (48 - 40 h)', fontdict=font)
#
# plt.subplot(3, 4, 4)
# plt.plot(weather[:,3], color='#6495ED', label=u'Wind direction')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("Wind direction($°$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(3, 4, 5)
# plt.plot(weather[:,4], color='#DC143C', label=u'Wind speed')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("Wind speed(km/h)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(3, 4, 6)
# plt.plot(weather[:,5],  color='#A9A9A9', label=u'Clouds')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("Clouds", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(3, 4, 7)
# plt.plot(weather[:,6], color='#a55af4', label=u'Maximum temperature')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("Maximum temperature($°$C)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(3, 4, 8)
# plt.plot(weather[:,7], color='#82cafc', label=u'Minimum temperature')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("Minimum temperature($°$C)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(3, 4, 9)
# plt.plot(weather[:,8], color='#ffdf22', label=u'Conditions')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("Conditions", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.show()

def sudden_changed(data):
    '''
    用于处理突变的值
    Args:
        city_dictionary:
    Returns:
'''
    shape=data.shape
    print(shape)
    for i in range(shape[0]):
        if i!=0:
            if data[i]-data[i-1]>100:
                data[i] = data[i - 1]
    return data

pollution=pd.read_csv(file+'train_around_weather.csv',usecols=['location','PM2.5'])

# plt.figure()
# # Label is observed value,Blue
# font = {'family': 'Times New Roman',
#         'weight': 'normal',
#         'size': 9,
#         }
# font1 = {'family': 'Times New Roman',
#         'weight': 'normal',
#         'size': 8,
#         }
#
# plt.subplot(4, 4, 1)
# plt.plot(pollution.loc[pollution['location']=='ShangHai'].values[:,1],color='#0cdc73', label=u'Shanghai')
# # use the legend
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# # plt.xlabel("time(hours)", fontsize=17,fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 1 (48 - 40 h)', fontdict=font)
# # plt.title("the prediction of PM2.5", fontsize=17,fontdict=font)
#
# plt.subplot(4, 4, 2)
# plt.plot(pollution.loc[pollution['location']=='NanJing'].values[:,1], color='#696969', label=u'Nanjing')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 2 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 3)
# plt.plot(pollution.loc[pollution['location']=='SuZhou'].values[:,1], color='#1E90FF', label=u'Suzhou')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 3 (48 - 40 h)', fontdict=font)
#
#
# a=sudden_changed(pollution.loc[pollution['location']=='NanTong'].values[:,1])
# plt.subplot(4, 4, 4)
# plt.plot(a, color='#228B22', label=u'Nantong')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 5)
# plt.plot(pollution.loc[pollution['location']=='WuXi'].values[:,1], color='#FF00FF', label=u'Wuxi')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 6)
# plt.plot(pollution.loc[pollution['location']=='ChangZhou'].values[:,1],  color='#FFD700', label=u'Changzhou')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 7)
# plt.plot(pollution.loc[pollution['location']=='ZhenJiang'].values[:,1], color='#FF69B4', label=u'Zhenjiang')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 8)
# plt.plot(pollution.loc[pollution['location']=='HangZhou'].values[:,1], color='#CD5C5C', label=u'Hangzhou')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 9)
# plt.plot(pollution.loc[pollution['location']=='NingBo'].values[:,1], color='#9370DB', label=u'Ningbo')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 10)
# plt.plot(pollution.loc[pollution['location']=='ShaoXing'].values[:,1], color='#0000CD', label=u'Shaoxing')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 11)
# plt.plot(pollution.loc[pollution['location']=='HuZhou'].values[:,1], color='#ADD8E6', label=u'Huzhou')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 12)
# plt.plot(pollution.loc[pollution['location']=='JiaXing'].values[:,1], color='#FFB6C1', label=u'Jiaxing')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 13)
# plt.plot(pollution.loc[pollution['location']=='TaiZhou'].values[:,1], color='#FFA07A', label=u'Taizhou')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.subplot(4, 4, 14)
# plt.plot(pollution.loc[pollution['location']=='ZhouShan'].values[:,1], color='#20B2AA', label=u'Zhoushan')
# plt.legend(loc='upper right',prop=font1)
# plt.grid(axis='y', linestyle='--')
# plt.xlabel("Time (hours)", fontdict=font)
# plt.ylabel("PM$_{2.5}$(ug/m$^3$)", fontdict=font)
# # plt.title('sample 4 (48 - 40 h)', fontdict=font)
#
# plt.show()