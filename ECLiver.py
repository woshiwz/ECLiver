# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:11:48 2026

@author: 10191
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:56:45 2023

@author: qiubinxu
"""

import streamlit as st
import pandas as pd #处理数据所用库
import numpy as np
import requests #工具，访问服务器
import numpy as np#加载数据所用库
import pandas as pd #处理数据所用库
import xgboost
import xgboost as xgb#极限梯度提升机所用库
from xgboost import XGBClassifier#分类算法#加载极限梯度提升机中分类算法
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree #导入需要的模块
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split#模型选择将数据集分为测试集和训练集
from sklearn.metrics import accuracy_score#模型最终的预测准确度分数
import matplotlib#加载绘图工具
from xgboost import plot_importance#加载极限梯度提升机中重要性排序函数
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score   # 准确率
import scipy.stats as stats
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score as CVS
from xgboost import plot_importance
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from IPython.display import display
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix
import eli5
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from eli5.sklearn import PermutationImportance
from IPython.display import display, Image
from sklearn.model_selection import cross_val_score,StratifiedKFold
st.title('Application of Machine Learning Methods to Predict Liver Metastases in Esophageal Cancer Patients'.center(33, '-'))
classes = {0:'NLM',1:'LM'}
st.sidebar.expander('')     
st.sidebar.subheader('Variable')       
Age=st.sidebar.selectbox('Age', ['<50','≥50'])
Age_map = {'<50':0,'≥50':1}
Sex=st.sidebar.selectbox('Sex', ['Male','Female'])
Sex_map = {'Male':0,'Female':1}
First=st.sidebar.selectbox('First', ['No','Yes'])
First_map = {'No':0,'Yes':1}
Grade=st.sidebar.selectbox('Grade',['Grade I','Grade II','Grade III',"Grade IV"])
Grade_map = {'Grade I':0,'Grade II':1,'Grade III':2,"Grade IV":3}
Radiology=st.sidebar.selectbox('Radiology', ['No','Yes'])
Radiology_map = {'No':0,'Yes':1}
N_stage=st.sidebar.selectbox('Nstage',['N0-1','N2-3'])
N_stage_map = {'N0-1':0,'N2-3':1}
Surgery=st.sidebar.selectbox("Surgery",['No','Yes'])
Surgery_map={'No':0,'Yes':1}
filename = 'modelECLiver.json'
x = []
x.extend([Age_map[Age],Sex_map[Sex],
         First_map[First],Grade_map[Grade],Radiology_map[Radiology],
         N_stage_map[N_stage],Surgery_map[Surgery]])
x=np.array(x).reshape(1,7)
import pickle
if st.button("Predict"):
    #  predict_class()
    import os
    if os.path.exists(filename):
        with open(filename, 'rb') as fq:
            modelXGB = pickle.load(fq, encoding='bytes')
            y_pred = modelXGB.predict_proba(x)
            print(max(y_pred[:,1]))
            st.header('Probability of liver metastases: %.2f %%' % (max(y_pred[:,1])* 100))