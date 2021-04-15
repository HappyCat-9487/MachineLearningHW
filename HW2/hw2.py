# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:28:01 2019

@author: halu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import random
from sklearn import metrics 
#導入資料
data=pd.read_csv('2ring.csv',header=None)
#分配資料
x1, x2, y=data.iloc[0:300, 0].values, data.iloc[0:300, 1].values, data.iloc[0:300, 2].values
#分配train資料集及test資料集
train_x1, test_x1, train_y, test_y = train_test_split(x1, y, test_size=0.25, random_state=0)
train_x2, test_x2, train_y, test_y = train_test_split(x2, y, test_size=0.25, random_state=0)

for j in range(len(train_y)):
    if(train_y[j]==2):
        train_y[j]=0
for j in range(len(test_y)):
    if(test_y[j]==2):
        test_y[j]=0
#權重值、偏差值、學習率、迴圈次數設定
w1=random.randint(-50,50)
w2=random.randint(-50,50)
b=random.randint(-50,0)
lr=0.001
epoch=100
k=np.zeros(epoch)       #裝填1000個average cross entropy
epoch_set=[]            #作為iteration顯示設定

#做logistic regression
for i in range(epoch):
    epoch_set.append(i+1)
    w1_gcross=0
    w2_gcross=0
    b_gcross=0
    for j in range(len(train_x1)):
        w1_gcross=w1_gcross-(train_y[j]-(1/(1+math.exp(w1*train_x1[j]+w2*train_x2[j]+b))))*train_x1[j]
        w2_gcross=w2_gcross-(train_y[j]-(1/(1+math.exp(w1*train_x1[j]+w2*train_x2[j]+b))))*train_x2[j]
        b_gcross=w2_gcross-(train_y[j]-(1/(1+math.exp(w1*train_x1[j]+w2*train_x2[j]+b))))*1
    w1 = w1-lr*w1_gcross
    w2 = w2-lr*w2_gcross
    b = b-lr*b_gcross
    cross_entropy=0
    for j in range(len(train_x1)):
        cross_entropy=(cross_entropy)-train_y[j]*math.log(1/(1+math.exp(w1*train_x1[j]+w2*train_x2[j]+b)))-(1-train_y[j])*(1-math.log(1/(1+math.exp(w1*train_x1[j]+w2*train_x2[j]+b))))
        aver_cross_entropy=(cross_entropy)/(len(train_x1))
        print(aver_cross_entropy)
        k[i]=aver_cross_entropy
        i+1
    

#最終的權重及偏差值
print()
print("weight1=",w1)
print("weight2=",w2)
print("bias=",b)
#每次的cross entropy數值
plt.plot(epoch_set,k,color='purple')
plt.xlabel('iteration',fontsize=18)
plt.ylabel('cross entropy',fontsize=18)
plt.show()

#train的圖示
for j in range(len(train_x1)):
    if(train_y[j]==1):
        plt.scatter(train_x1[j],train_x2[j],c='b',marker='o')       #分別class
    else:
        plt.scatter(train_x1[j],train_x2[j],c='r',marker='x')
   
l=np.linspace(-4,4,100)  
r, t=-w1/w2, -b/w2
plt.plot(l, r*l+t, color='black')           #畫出機率=0.5的那條線
plt.xlabel('train_x1',fontsize=18)
plt.ylabel('train_x2',fontsize=18)
train_y_predict=np.zeros(len(train_x1))
for j in range(len(train_x1)):
    train_y_predict[j]=1/(1+math.exp(w1*train_x1[j]+w2*train_x2[j]+b))
    if(train_y_predict[j]>=0.5):
        train_y_predict[j]=1
    else:
        train_y_predict[j]=0
    
accuracy = metrics.accuracy_score(train_y,train_y_predict)          #計算training辨識率
accuracy = accuracy*100
plt.text(-3,-2,'%.2f percent' % accuracy, fontdict={'size': 20, 'color':  'green'})   
plt.show()

#計算testing cross entropy 
cross_entropy_1=0
for j in range(len(test_x1)):
    cross_entropy_1=(cross_entropy_1)-test_y[j]*math.log(1/(1+math.exp(w1*test_x1[j]+w2*test_x2[j]+b)))-(1-test_y[j])*(1-math.log(1/(1+math.exp(w1*test_x1[j]+w2*test_x2[j]+b))))
    aver_cross_entropy_1=(cross_entropy_1)/(len(test_x1))

#test的圖示
for j in range(len(test_y)):
    if(test_y[j]==1):
        plt.scatter(test_x1[j],test_x2[j],c='b',marker='o')
    else:
        plt.scatter(test_x1[j],test_x2[j],c='r',marker='x')
     
l=np.linspace(-4,4,100)  
r, t=-w1/w2, -b/w2
plt.plot(l, r*l+t, color='black')               #畫出機率=0.5的那條線
plt.xlabel('test_x1',fontsize=18)
plt.ylabel('test_x2',fontsize=18)
test_y_predict=np.zeros(len(test_x1))
for j in range(len(test_x1)):
    test_y_predict[j]=1/(1+math.exp(w1*test_x1[j]+w2*test_x2[j]+b))
    if(test_y_predict[j]>=0.5):
        test_y_predict[j]=1
    else:
        test_y_predict[j]=0
    
accuracy1 = metrics.accuracy_score(test_y,test_y_predict)           #計算testing辨識率
accuracy1 = accuracy1*100
plt.text(-3,-2,'%.2f percent' % accuracy1, fontdict={'size': 20, 'color':  'green'})     
plt.show()

    
