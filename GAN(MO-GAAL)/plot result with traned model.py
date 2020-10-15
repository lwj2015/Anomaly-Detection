from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_data():
    #data = pd.read_table('Data/onecluster', sep=',', header=None)
    data = pd.read_excel('Data/result4_new_new.xls',  header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1)
    data_x = data.to_numpy()
    data_id = id.values
    data_y = y.values
    return data_x, data_y, data_id

data_x, data_y, data_id = load_data()

discriminator = load_model('MO_GAAL_Discriminator_of_zhusuji.h5')

p = discriminator.predict(data_x)

result = np.concatenate((data_y.reshape(-1,1),p.reshape(-1,1)), axis = 1)
result = pd.DataFrame(result,columns=['type','prediction'])

cnt = 0
for i in range(len(result['type'])):
    if result['prediction'][i] > 0.5 and result['type'][i] == 'nor' or result['prediction'][i] < 0.5 and result['type'][i] == 'out':
        print(str(i)+' '+str(result['type'][i])+ ' ' + str(result['prediction'][i]))
        cnt+=1
print(cnt)
#print(result)

result = result.to_numpy()
#print(result)
r = np.argsort(result[:,1])
#print(r)
#print(result[r])

pd.DataFrame(result[r]).to_excel("prediction score of zhusuji.xls")

re = result[r]

normal = []
abnormal = []

i = 0

for j in range(len(r)):
    if re[j][0] == 'nor' :
        normal.append([i, re[j][1]])
    else:
        abnormal.append([i, re[j][1]])
    i+=1

#plt.plot(range(len(r)), re[:,1],color='black')
axes = plt.subplot(111)

normal = np.array(normal)
abnormal = np.array(abnormal)
type1 = axes.scatter(normal[ : , 0],   normal[ : , 1],   s=10, color='red')
type2 = axes.scatter(abnormal[ : , 0], abnormal[ : , 1], s=12, color='blue')

axes.legend((type1, type2), ("norm", "anomaly") , prop={'size':20})
axes.set_xlabel('number')
axes.set_ylabel('score')

plt.title("MO-GAAL")
plt.show()

