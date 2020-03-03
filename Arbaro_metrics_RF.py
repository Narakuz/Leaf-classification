# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:56:11 2019

@author: ZhuX
"""

import pandas as pd
import numpy as np
#import random
from glob import glob 
import re
#import os
import multiprocessing 
from laspy.file import File
import scipy.special
import itertools
from CV_plot import CV_plot
from sklearn.model_selection import train_test_split
#from functools import partial
#from sklearn.neighbors import NearestNeighbors
#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score
import scipy
import matplotlib.pyplot as plt
#import vtk
#import vtk.util.numpy_support as vtk_np
#import statsmodels.api as sm
import statsmodels.formula.api as sm
#import seaborn as sns
from statsmodels.stats.outliers_influence import summary_table
import hdf5storage
from Validation import lineplotCI
from correlation import correlation
import pickle
import matplotlib 
import statsmodels.formula.api as sm
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.datasets import make_regression
from scipy.stats import binned_statistic_2d,linregress
from sklearn import metrics

def LAI(x):
    data = np.loadtxt(x, skiprows=1)    
    return np.sum(data[:,4])/(np.pi*50**2)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

### Arbaro 
fvalidations = glob('D:/Data/Personal/Zhux/helios_simu/plot_ALS/para*.txt',recursive=True)
fvalidations = fvalidations[0:len(fvalidations)]
result_list = list(map(LAI, fvalidations))
Y_arbaro = np.asarray(result_list)

Fusion_arbaro = pd.read_csv('//dikke.itc.utwente.nl/Homedir/zhux/Paper/6th/analysis/Arbaro_fusionmetrics.csv',index_col = 0)
Fusion_arbaro = Fusion_arbaro.sort_values(by='FileTitle')
Fusion_arbaro = Fusion_arbaro.drop('FileTitle',axis=1)
Fusion_arbaro.index = np.arange(1,31)
Cover_arbaro = pd.read_csv('//dikke.itc.utwente.nl/Homedir/zhux/Paper/6th/analysis/Arbaro_metrics_canopy.csv',index_col = 0)
Cover_arbaro.index = np.arange(1,31)
Metrics_arbaro = pd.concat([Fusion_arbaro,Cover_arbaro],axis=1)
Metrics_arbaro=Metrics_arbaro.drop(Metrics_arbaro.loc[:,'Int minimum':'Int P99'].head(0).columns, axis=1)

### BFNP
Fusion_BFNP = pd.read_csv('//dikke.itc.utwente.nl/Homedir/zhux/Paper/6th/analysis/BFNP_fusionmetrics.csv',index_col = 0)
Fusion_BFNP = Fusion_BFNP.sort_values(by='FileTitle')
Fusion_BFNP = Fusion_BFNP.drop('FileTitle',axis=1)
Fusion_BFNP.index = np.arange(1,37)
Cover_BFNP = pd.read_csv('//dikke.itc.utwente.nl/Homedir/zhux/Paper/6th/analysis/CC2016.csv',index_col = 0)
Cover_BFNP.index = np.arange(1,37)
Metrics_BFNP = pd.concat([Fusion_BFNP,Cover_BFNP],axis=1)
Metrics_BFNP = Metrics_BFNP.drop(Metrics_BFNP.loc[:,'Int minimum':'Int P99'].head(0).columns, axis=1)

TLS = pd.read_csv('//dikke.itc.utwente.nl/Homedir/zhux/Paper/6th/analysis/TLS_Joe_5d_G2_2017.csv',index_col = 0)
TLS.index = np.arange(1,37)


#### random forest BFNP
Y_BFNP = TLS.LAI.values
X = Metrics_BFNP.values
names = Metrics_BFNP.columns.values.tolist()
regr = RandomForestRegressor(n_estimators=500)
regr.fit(X, Y_BFNP)
importance = pd.DataFrame(index = names, columns = ['importance'],data = regr.feature_importances_)
importance = importance.sort_values(by=['importance'],ascending=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
f, ax = plt.subplots(figsize=(10, 5))
#plt.title('b',fontsize=15)
CI_df = pd.DataFrame(columns = ['x', 'Bias1', 'Bias2'])
0.1*importance.shape[0]

CI_df['x'] = importance[importance.importance>0.01].importance  
rects1 = ax.barh(importance[importance.importance>0.01].index, CI_df['x'],color='g',align='center')
matplotlib.rc('ytick', labelsize=10) 
matplotlib.rc('xtick', labelsize=10) 
#plt.xticks(rotation=70)
ax.grid(True)
ax.set_xlabel('RF_importance value',fontsize=14)
ax.set_ylabel('Metrics',fontsize=14)
plt.tight_layout()

Y = TLS.LAI.values
X = Metrics_BFNP.values

importance = importance.sort_values(by=['importance'],ascending=False)
X = Metrics_BFNP[importance[0:5].index].values
    
loo = model_selection.LeaveOneOut()
ytests = []
ypreds = []
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx] #requires arrays
    y_train, y_test = Y[train_idx], Y[test_idx]
    
    model = RandomForestRegressor(n_estimators=500)
#        model = linear_model.LassoLars(alpha=0.01,fit_intercept=True,
#         fit_path=True, max_iter=500, normalize=True, positive=False,
#         precompute='auto', verbose=False)
#        model = SVR(kernel="linear")
#        model = PLSRegression(n_components=5)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)        
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.        
    ytests += list(y_test)
    ypreds += list(y_pred)
ytests = np.asarray(ytests)
ypreds = np.asarray(ypreds)    
CV_plot(np.asarray(ytests),np.asarray(ypreds),'Observed eLAI','Measured eLAI','',' ')
plt.axis([0,3,0,3])

#### random forest Arbaro
Y_arbaro = Y_arbaro
X = Metrics_arbaro.values
names = Metrics_arbaro.columns.values.tolist()
regr = RandomForestRegressor(n_estimators=500)
regr.fit(X, Y_arbaro)
importance = pd.DataFrame(index = names, columns = ['importance'],data = regr.feature_importances_)
importance = importance.sort_values(by=['importance'],ascending=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
f, ax = plt.subplots(figsize=(10, 5))
#plt.title('b',fontsize=15)
CI_df = pd.DataFrame(columns = ['x', 'Bias1', 'Bias2'])
0.1*importance.shape[0]

CI_df['x'] = importance[importance.importance>0.01].importance  
rects1 = ax.barh(importance[importance.importance>0.01].index, CI_df['x'],color='g',align='center')
matplotlib.rc('ytick', labelsize=10) 
matplotlib.rc('xtick', labelsize=10) 
#plt.xticks(rotation=70)
ax.grid(True)
ax.set_xlabel('RF_importance value',fontsize=14)
ax.set_ylabel('Metrics',fontsize=14)
plt.tight_layout()

Y = Y_arbaro
X = Metrics_arbaro.values
importance = importance.sort_values(by=['importance'],ascending=False)
X = Metrics_arbaro[importance[0:5].index].values
    
loo = model_selection.LeaveOneOut()
ytests = []
ypreds = []
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx] #requires arrays
    y_train, y_test = Y[train_idx], Y[test_idx]
    
    model = RandomForestRegressor(n_estimators=500)
#        model = linear_model.LassoLars(alpha=0.01,fit_intercept=True,
#         fit_path=True, max_iter=500, normalize=True, positive=False,
#         precompute='auto', verbose=False)
#        model = SVR(kernel="linear")
#        model = PLSRegression(n_components=5)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)        
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.        
    ytests += list(y_test)
    ypreds += list(y_pred)
ytests = np.asarray(ytests)
ypreds = np.asarray(ypreds)    
CV_plot(np.asarray(ytests),np.asarray(ypreds),'Observed eLAI','Measured eLAI','',' ')
plt.axis([0,3,0,3])



#### training with Arbaro, testing on BFNP
#### random forest Arbaro
Y_arbaro = Y_arbaro
X = Metrics_arbaro.values
names = Metrics_arbaro.columns.values.tolist()
regr = RandomForestRegressor(n_estimators=500)
regr.fit(X, Y_arbaro)
importance = pd.DataFrame(index = names, columns = ['importance'],data = regr.feature_importances_)
importance = importance.sort_values(by=['importance'],ascending=True)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
f, ax = plt.subplots(figsize=(10, 5))
#plt.title('b',fontsize=15)
CI_df = pd.DataFrame(columns = ['x', 'Bias1', 'Bias2'])
CI_df['x'] = importance[importance.importance>0.01].importance  
rects1 = ax.barh(importance[importance.importance>0.01].index, CI_df['x'],color='g',align='center')
matplotlib.rc('ytick', labelsize=10) 
matplotlib.rc('xtick', labelsize=10) 
#plt.xticks(rotation=70)
ax.grid(True)
ax.set_xlabel('RF_importance value',fontsize=14)
ax.set_ylabel('Metrics',fontsize=14)
plt.tight_layout()
importance = importance.sort_values(by=['importance'],ascending=False)
X = Metrics_arbaro[importance[importance.importance>0.01].index].values
regr = RandomForestRegressor(n_estimators=500)
regr.fit(X, Y_arbaro)

X_test = Metrics_BFNP.values
y_test = TLS.LAI.values

X_test = Metrics_BFNP[importance[importance.importance>0.01].index].values    
y_predict = regr.predict(X_test)
lineplotCI(y_test,y_predict,'eLAI from TLS','eLAI from ALS','','')
plt.axis([0,10,0,10])
