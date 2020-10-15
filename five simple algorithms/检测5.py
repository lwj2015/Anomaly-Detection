import xlwt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

# 需要有正确训练集，因为one-class的本质就是只指导正确的来发现错误的

Datasets = pd.read_excel(r'D:\PyCharm_Community_Edition\gru\datasets\time.xls').values.astype('float32')
x = Datasets[1: , 0]
y = Datasets[1: , 1]

# 选出正确的数据
Datasets2 =pd.read_excel(r'D:\PyCharm_Community_Edition\gru\datasets\result4.xls').values
y2 = Datasets2[:-1 , 1][Datasets2[:-1 , 2] == 'yes']

# 防止过拟合我采用随机选取其中的一半来训练
index = np.random.randint(len(y2) , size = (len(y2)//2,))
y_train = y2[index]

y_train_x = np.ones((len(y_train),2))
y_x = np.ones((len(y),2))

for i in range(len(y_train)):
    y_train_x[i][1] = y_train[i]
for i in range(len(y_x)):
    y_x[i][1] = y[i]

print(y_train_x)


model_isof = svm.OneClassSVM(nu=0.85, kernel="rbf", gamma=0.01)
model_isof.fit(y_train_x)
y_pred = model_isof.predict(y_x)

print(y_train)
print(y)

normal = []
abnormal = []

print(y_pred)

for j in range(len(y_pred)):
    if y_pred[j] == 1 :
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

plt.title("One class SVM")
plt.show()

book = xlwt.Workbook()          # 新建工作簿
sheet = book.add_sheet('Test')  # 添加工作页

for i in range(len(x)):
        sheet.write(i, 0, int(x[i]))
        sheet.write(i, 1, int(y[i]))
        if y_pred[i] == -1 :
            sheet.write(i, 2, 'yes')
        else:
            sheet.write(i, 2, 'no')

book.save(filename_or_stream='result5.xls') # 一定要保存
