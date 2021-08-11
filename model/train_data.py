# -*- coding: utf-8 -*-

import pandas as pd

def get_data(file_path):
    '''
    :param file_path:
    :return:
    '''
    data=pd.read_csv(file_path,encoding='utf-8')
    return data.values[:,2:]