# -- coding: utf-8 --

import xlrd
from datetime import date, datetime

file='/Users/guojianzou/Downloads/weather'

col_name = ['time', 'temperature', 'humidity', 'pressure', 'wind_direction', 'wind_speed',
            'clouds', 'maximum_temperature', 'minimum_temperature', 'conditions']


def read_excel(file_name,condition_type,conditions_exit):
    pre_values = {'time': 0, 'temperature': 0, 'humidity': 0, 'pressure': 0, 'wind_direction': 0, 'wind_speed': 0,
                  'clouds': 0, 'maximum_temperature': 0, 'minimum_temperature': 0, 'conditions': 0}
    # 打开文件
    workbook = xlrd.open_workbook(file_name)
    # 获取所有sheet
    sheet_name = workbook.sheet_names()[0]

    # 根据sheet索引或者名称获取sheet内容
    sheet = workbook.sheet_by_index(0)  # sheet索引从0开始
    # sheet = workbook.sheet_by_name('Sheet1')

    # print (workboot.sheets()[0])
    # sheet的名称，行数，列数
    print(sheet.name, sheet.nrows, sheet.ncols)

    # 获取整行和整列的值（数组）
    rows = sheet.row_values(0)  # 获取第1行内容
    # cols = sheet.col_values(2) # 获取第3列内容
    # print(rows)
    array=[]
    for rown in range(sheet.nrows-1, 0, -1):
        line=sheet.row_values(rown)
        # print(line)
        time = line[0].split('/')
        time.reverse()
        time='-'.join(time)+'-'+line[1][:2]

        temperature=float(line[2])

        humidity=float(pre_values['humidity'] if line[3]=='' else line[3].split('%')[0])
        pre_values['humidity']=humidity

        pressure=float(pre_values['pressure'] if line[4]=='-' else line[4].split('Hpa')[0])
        pre_values['pressure']=pressure

        wind_direction=float( -1 if line[5]=='calm.' else line[5].split('º')[0])

        wind_speed=float(pre_values['wind_speed'] if line[6]=='' else line[6])
        pre_values['wind_speed']=wind_speed

        print(line[7])
        clouds=float( -1 if line[7][0]=='N' or line[7]=='-' else line[7].split('/')[0])

        maximum_temperature=float(temperature if line[12]=='-' else line[12])

        minimum_temperature=float(temperature if line[13]=='-' else line[13])

        if line[14] not in conditions_exit:
            condition_type +=1
            conditions_exit[line[14]]=condition_type
        conditions=conditions_exit[line[14]]

        print(time,temperature,humidity,pressure,wind_direction,wind_speed,clouds,maximum_temperature,minimum_temperature,conditions)
        array.append([time,temperature,humidity,pressure,wind_direction,wind_speed,clouds,maximum_temperature,minimum_temperature,conditions])
        # array = rows
        # array['L1'] = sheet.cell_value(rown, 0)
        # array['L2'] = sheet.cell_value(rown, 1)
        # array['L3'] = sheet.cell_value(rown, 2)
        # array['L4'] = sheet.cell_value(rown, 3)
        # array['Question'] = sheet.cell_value(rown, 4)
        # array['Answer'] = sheet.cell_value(rown, 5)
        # tables.append(array)
    return array

def write(csv_writer,data):
    for line in data:
        csv_writer.writerow(line)

import csv
if __name__ == '__main__':
    # 读取Excel
    condition_type = 0
    conditions_exit = {'-': condition_type}

    # 1. 创建文件对象
    f = open(file+'/weather.csv', 'w', encoding='utf-8')

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 3. 构建列表头
    csv_writer.writerow(col_name)


    for i in range(1,13):
        data=read_excel(file+'/'+str(i)+'.xls',condition_type,conditions_exit)
        # 4. 写入csv文件内容
        write(csv_writer,data)

    # 5. 关闭文件
    f.close()

    print('读取成功 和 写入成功！！！')
