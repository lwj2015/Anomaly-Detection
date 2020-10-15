import xlwt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Datasets = pd.read_excel('time.xls').values.astype('float32')
x = Datasets[1: , 0]
y = Datasets[1: , 1]

random_state = 10
y_pred = KMeans(n_clusters=5, init='random',random_state=random_state).fit_predict(y.reshape(-1,1))

print(y_pred)

normal = []
abnormal = []

for j in range(len(y_pred)):
    if y_pred[j] < 1 :
        normal.append([x[j], y[j]])
    else:
        abnormal.append([x[j], y[j]])

plt.plot(x,y,color='black')
axes = plt.subplot(111)

normal = np.array(normal)
abnormal = np.array(abnormal)
type1 = axes.scatter(normal[ : , 0],   normal[ : , 1],   s=10, color='red')
type2 = axes.scatter(abnormal[ : , 0], abnormal[ : , 1], s=12, color='blue')

axes.legend((type1, type2), ("norm", "anomaly") , prop={'size':20})

plt.title("k-means")
plt.show()

book = xlwt.Workbook()          # 新建工作簿
sheet = book.add_sheet('Test')  # 添加工作页

for i in range(len(x)):
        sheet.write(i, 0, int(x[i]))
        sheet.write(i, 1, int(y[i]))
        if y_pred[i] < 1:
            sheet.write(i, 2, 'yes')
        else:
            sheet.write(i, 2, 'no')

book.save(filename_or_stream='result2.xls') # 一定要保存