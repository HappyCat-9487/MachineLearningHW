# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:11:13 2019

@author: halu
"""

import random
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('pokemon.csv')

x, y = dataset.iloc[1:76, 2].values, dataset.iloc[1:76, 3].values
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=0)



w = random.randint(-100,100) 
b = random.randint(-110,0) 
alpha = 0.0000000003   
epoch = 5000 
k=np.zeros(epoch)
epoch_set=[]
for i in range(epoch):
    epoch_set.append(i+1)

# train
for i in range(epoch):
    
    y = w * train_x + b
    
    w1_grad = 0
    b_grad = 0
    for j in range(len(train_x)):
        w1_grad = w1_grad - 2*(train_y[j] - b - w * train_x[j])*train_x[j]
        b_grad = b_grad - 2*(train_y[j] - b - w * train_x[j])
    
    w = w - alpha * w1_grad   
    b = b - alpha * b_grad

    trainloss = ((train_y - y)**2)**0.5
    loss = sum(trainloss) / (len(train_x))
    print(loss)
    k[i]=loss
    
#顯示權重weight及偏差值bias
print()
print("weight=",w)
print("bias",b)
#畫loss Func 的形成圖
plt.plot(epoch_set,k,color='purple')
plt.xlabel('iteration',fontsize=18)
plt.ylabel('Loss',fontsize=18)
plt.show()        
#畫training圖
plt.scatter(train_x, train_y, color='blue')
plt.plot(train_x, y, 'r-')
plt.text(0, 1500, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color':  'red'})
plt.xlabel('train_X',fontsize=18)
plt.ylabel('train_y',fontsize=18)        
plt.show()

#計算testing loss
y1 = w * test_x + b
testloss = ((test_y - y1)**2)**0.5
loss1 = sum(testloss) / (len(test_x))

#畫testing圖
plt.scatter(test_x, test_y, color='green')
plt.plot(test_x, y1, 'r-')
plt.text(0, 550, 'Loss=%.4f' % loss1, fontdict={'size': 20, 'color': 'red'})
plt.xlabel('test_x',fontsize=18)
plt.ylabel('test_y',fontsize=18)
plt.show()