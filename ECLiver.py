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
import numpy as np
import xgboost as xgb
import os
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
