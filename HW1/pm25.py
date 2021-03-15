# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 01:11:58 2019

@author: User
"""

# -*- coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
#讀取csv資料檔案
data=pd.read_csv("pm25traindata.csv")
data1=pd.read_csv("pm25testdata.csv")

pm2_5=data[data['測項']=='PM2.5'].ix[:,3:]
testpm2_5=np.array(data1.ix[:,2:])

train_data = np.array(pm2_5)
test_data = np.array(data1)

#train的xy資料建置
#計算每日PM2.5資料均值
for m in range(len(train_data)):
    for n in range(0,24):
        if type(train_data[m][n]) == float:
            train_data[m][n] = 0.0
            continue
        else:  
            train_data[m][n] = int(train_data[m][n])
            
a=[]
for m in range(len(train_data)):
    count=0
    total=0
    for n in range(0,24): 
        if train_data[m][n] != 0.0:
            count = count+1
            total = total+ train_data[m][n] 
    a.append(total/count)

#建立y=wx+b中的xy資料
b=a

index = [131]
index1 = [0]
train_x = np.delete(a, index)
train_y = np.delete(b, index1)

#test的xy資料建置
#取test資料中PM2.5在陣列中的位置
place=[]
place=(np.where(test_data == 'PM2.5')[0])

test_data=[]
for i in range(len(place)):
    test_data.append(testpm2_5[place[i]])

#計算每日PM2.5資料均值
for m in range(len(test_data)):
    for n in range(0,24):
        if type(test_data[m][n]) == float:
            test_data[m][n] = 0.0
            continue
        else:  
            test_data[m][n] = int(test_data[m][n])            
                               
place_1=[]
for m in range(len(place)):
    count=0
    total=0
    for n in range(0,24): 
        if test_data[m][n] != 0.0:
            count = count+1
            total = total+ test_data[m][n] 
    place_1.append(total/count)    

#建立y=wx+b中的xy資料
place_new=place_1

index = [119]
index1 = [0]
test_x = np.delete(place_1, index)
test_y = np.delete(place_new, index1)

#regression
b=random.randint(-150,-100)
w=random.randint(-5,5)
lr=0.00001
iteration=8000
loss_set=[]

#gradient
for i in range(iteration):
    b_grad=0   # 新的b點位移預測
    w_grad=0   # 新的w點位移預測
    
    for n in range(len(train_x)):

        b_grad = b_grad -2.0* (train_y[n] - b - w*train_x[n]) * 1.0
        w_grad = w_grad -2.0* (train_y[n] - b - w*train_x[n]) * train_x[n]
        
    b = b - lr * b_grad 
    w = w - lr * w_grad
    
    bb = 0.0
    for n in range(len(train_x)):
        bb = ((train_y[n]-(b+w*train_x[n]))**2)**0.5 + bb  
    loss_set.append(bb/len(train_x))  
    
iteration_set=[]
for i in range(0,iteration):
    iteration_set.append(i+1)   
    
print("bias=",b)
print("weight=",w) 

#計算loss
aa=0.0
bb=0.0
dd=[]

#training loss
for n in range(len(train_x)):
    bb =((train_y[n]-(b+w*train_x[n]))**2)**0.5 + bb 

#testing loss
for i in range(len(test_x)):
    aa = ((test_y[i]-(b+w*test_x[i]))**2)**0.5 + aa     

print("training loss=",bb/len(train_x))    
print("testing loss=",aa/len(test_x))

#畫圖    
#畫y = w * x + b方程式
for m in range(len(train_x)):    
    dd.append(b+w*train_x[m]) 

#畫座標
plt.plot(iteration_set,loss_set)
plt.xlabel(r'$Iteration$', fontsize=16)
plt.ylabel(r'$Loss$', fontsize=16)

plt.figure()
plt.scatter(train_x,train_y,c="b")
plt.plot(train_x,dd,c="r")
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$y$', fontsize=16)

plt.figure()
plt.scatter(test_x,test_y,c="b")
plt.plot(train_x,dd,c="r")
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$y$', fontsize=16)

plt.show()