import xlwt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

Datasets = pd.read_excel('time.xls').values.astype('float32')
x = Datasets[1: , 0]
y = Datasets[1: , 1]

y_normalize = normalize(y.reshape(-1,1), axis = 0)
y_mean = y_normalize.mean()
y_std = y_normalize.std()

bool = np.zeros_like(y)

for i in range(len(y_normalize)):
    if y_normalize[i][0] - y_mean > 3 * y_std:
        bool[i] = 1

subplots = 1
size = len(x)//subplots

for i in range(subplots):
    normal = []
    abnormal = []

    for j in range(size * i ,size * (i+1)):
        if bool[j] == 0:
            normal.append([ x[j] , y[j] ])
        else:
            abnormal.append([x[j], y[j]])

    normal = np.array(normal)
    abnormal = np.array(abnormal)

    plt.figure(figsize=(8, 5))
    plt.title("3∂")
    plt.plot(x[size * i:size * (i + 1)], y[size * i:size * (i + 1)], color='black')
    axes = plt.subplot(111)

    type1 = axes.scatter(normal[ : , 0],   normal[ : , 1],   s=10, color='red')
    type2 = axes.scatter(abnormal[ : , 0], abnormal[ : , 1], s=12, color='blue')

    axes.legend((type1, type2), ("norm", "anomaly") , prop={'size':20})

    plt.show()

book = xlwt.Workbook()          # 新建工作簿
sheet = book.add_sheet('Test')  # 添加工作页

for i in range(len(x)):
        sheet.write(i, 0, int(x[i]))
        sheet.write(i, 1, int(y[i]))
        if bool[i] == 0:
            sheet.write(i, 2, 'yes')
        else:
            sheet.write(i, 2, 'no')

book.save(filename_or_stream='result1.xls') # 一定要保存



