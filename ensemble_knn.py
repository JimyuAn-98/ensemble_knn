# -*- coding: utf-8 -*-     支持文件中出现中文字符
###################################################################################################################

"""
Created on Thu Nov 12 20:15:02 2020

@author: Huangjiyuan

代码功能描述: （1）读取处理后的数据
            （2）使用基于KNN的组合分类器完成分类的运算，并输出准确度和混淆矩阵

"""
###################################################################################################################

import pandas as pd
import numpy as np
import math
import sklearn.model_selection as ms                #用于划分训练集和测试集
from sklearn.neighbors import KNeighborsClassifier  #KNN分类器
from sklearn.ensemble import BaggingClassifier      #Bagging分类器
from sklearn.metrics import confusion_matrix        #用于输出混淆矩阵

#读取文件并将两个文件合并
columns_name = ['mean','var','dwt_appro','dwt_detail','sampen','hurst','pfd']               #设置由各个属性组成的矩阵

dt113 = pd.read_excel(r'result_%d.xlsx'% (113))                                             #读取之前处理得到的result_113文件
dt113 = dt113.iloc[:,1:]                                                                    #去除掉第一列，也就是表格中的序列列
dt113.columns = ['label','mean','var','dwt_appro','dwt_detail','sampen','hurst','pfd']      #为每一列添加名称

dt114 = pd.read_excel(r'result_%d.xlsx'% (114))                                             #读取之前处理得到的result_114文件
dt114 = dt114.iloc[:,1:]
dt114.columns = ['label','mean','var','dwt_appro','dwt_detail','sampen','hurst','pfd']

dt = pd.concat([dt113,dt114],axis=0,ignore_index=True)                                      #将两个矩阵合并成一个矩阵

x = np.array(dt[columns_name])  #得到仅有属性值的矩阵，作为分类运算的输入
y = np.array(dt['label'])       #得到仅有标签的矩阵，作为分类运算的输出

X_train, X_test, Y_train, Y_test = ms.train_test_split(x,y,test_size=0.2,random_state=7)    #划分出训练集与测试集

#基于KNN的组合分类器
for k in range(1,11):
    clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=k),n_estimators=9)
    clf.fit(X_train, Y_train.ravel())
    #获得准确率、混淆矩阵
    # 准确率
    print('k=%d时，测试集准确率为：'%k,clf.score(X_test, Y_test))
    # 混淆矩阵
    y_pred = clf.predict(X_test)
    m = confusion_matrix(Y_test, y_pred)
    print('k=%d时，混淆矩阵为\n'%k,m)
