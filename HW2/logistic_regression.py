# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:13:46 2019

@author: pc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:13:17 2019

@author: user01
"""
import math
import random
import matplotlib.pyplot as plt #畫圖用函數庫
import numpy as np
import pandas as pd

def sigmoid(scores):
    return 1/(1+np.exp(-scores))

def log_likelihood(features,target,weights):
    scores=np.dot(features,weights)
    l1=np.sum(target*scores/100-np.log(1+np.exp(scores/100)))
    #print(int(scores[0]))
    #print(np.exp(scores[0]/1000))
    return l1

def logistic_regression(features,target,iterations,learning_rate):
    make_plot=[]
    #代表如果是否需要截距項 True的話就會建立之
    intercept=np.ones((features.shape[0],1))
    features=np.hstack((intercept,features))
    
    weights=np.zeros(features.shape[1])
    #開始進行迴圈跑最出最佳參數
    for step in range(iterations):
        scores=np.dot(features,weights)
        #function套入sigmoid function得1機率值
        predictions=sigmoid(scores)
        #觀看誤差
        error=target-predictions
                     
        gradient=np.dot(features.T,error)
        
        weights+=learning_rate*gradient
        
        #print(scores)
        #print(features)
        
        #print(target)
        #print(predictions)
        #print(error)  
        #gradient
        #print(gradient)
        #print(gradient)
        print(weights)
        
        #print出藉由參數不斷的調整 Loss function不斷在向最小化邁進
        if step%1==0:
            make_plot.append(log_likelihood(features,target,weights))
    return weights,make_plot

#讀資料
data = pd.read_csv(r'D:\資料集改\2cring.csv',header=None)
data = np.array(data)
#資料隨機取樣
a=[]
b=[]
c=[]
test_a=[]
test_b=[]
test_c=[]
d=[]
e=[]
temp=data.shape[0]
for k in range (0,temp):
    d.append(k)    
e = random.sample(d,int(temp*2/3))
d1=set(d)
e1=set(e)

result=list(d1.difference(e1))

for i in range (0,int(temp*2/3)):    
    a.append(data[e[i]][0])
    b.append(data[e[i]][1])
    c.append(data[e[i]][2])
for k in range (0,temp-(int(temp*2/3))):    
    test_a.append(data[result[k]][0])
    test_b.append(data[result[k]][1])
    test_c.append(data[result[k]][2])
count=0
data_w = []*2
dataset = [data_w]*int(temp*2/3)
while (count < int(temp*2/3)):
    if c[count] == 2:
        c[count] = 0
    dataset[count] = ([a[count],b[count]],[c[count]])
    count += 1
x1=[]
x2=[]
i=0
iterations=100
for x in dataset:
    x1.append(x[0])
    x2.append(x[1][0])
x1=np.array(x1)
x2=np.array(x2)
x2=x2.T

test_temp=temp-(int(temp*2/3))

count=0
data_w = []*2
test_dataset = [data_w]*(test_temp)
while (count < test_temp):
    if test_c[count] == 2:
        test_c[count] = 0
    test_dataset[count] = ([test_a[count],test_b[count]],[test_c[count]])
    count += 1
    
test_x1=[]
test_x2=[]
i=0
for x in test_dataset:
    test_x1.append(x[0])
    test_x2.append(x[1][0])
test_x1=np.array(test_x1)
test_x2=np.array(test_x2)
test_x2=test_x2.T

weights,make_plot=logistic_regression(x1,x2,iterations,learning_rate=0.5)
#print(make_plot)

#training accuracy
intercept=np.ones((x1.shape[0],1))
features=np.hstack((intercept,x1))
scores=np.dot(features,weights)
#function套入sigmoid function得1機率值
predictions=sigmoid(scores)
for i in range(len(predictions)):
    predictions[i] = int(predictions[i])
error=x2-predictions

count = 0
for i in range(int(temp*2/3)):
    if error[i] == 0:
        count = count + 1
train_accuracy = count / int(temp*2/3)

#testing accuracy
intercept=np.ones((test_x1.shape[0],1))
features=np.hstack((intercept,test_x1))
scores=np.dot(features,weights)
#function套入sigmoid function得1機率值
test_predictions=sigmoid(scores)
for i in range(len(test_predictions)):
    test_predictions[i] = int(test_predictions[i])
test_error=test_x2-test_predictions
count = 0
for i in range(test_temp):
    if test_error[i] == 0:
        count = count + 1
test_accuracy = count / test_temp 

print(weights)
print(train_accuracy)
print(test_accuracy)
plt.plot(make_plot)

plt.figure()
plt.scatter(a,b,c=predictions)

plt.figure()
plt.scatter(test_a,test_b,c=test_predictions)
#plt.scatter(np.hstack((a,test_a)),np.hstack((b,test_b)),c=np.hstack((predictions,test_predictions)))