# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:53:11 2021

@author: Saber
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
from helper_funcs import sumOfSines, RK4Step, mat_mul_series, Run_Data
import pickle
import seaborn as sns
from sklearn import preprocessing, linear_model, svm, decomposition

with open('optimal_runs_dt5ms_p1_pinv2_loss.pickle', 'rb') as f:
    optimal_runs = pickle.load(f)
    
k_data_ts = np.array([elem.controllers.squeeze() 
                      for elem in optimal_runs])
k_labels = np.array([elem.loss
                     for elem in optimal_runs])
k_data = np.median(k_data_ts, axis=-1)
# Get rid of the tiny last dimension
k_data = k_data[:,:4]

data_scaler0 = preprocessing.StandardScaler().fit(k_data)
k_data_scaled = data_scaler0.transform(k_data)
# 2D representation in error-effort space

# Linear regression 
mapper0 = linear_model.LinearRegression().fit(k_data, k_labels)
print(mapper0.score(k_data, k_labels))

# mapper1 = svm.SVR(kernel='rbf').fit(k_data, k_labels)
# print(mapper1.score(k_data, k_labels))



# # Unsupervised Dimensionality reduction
mapper_pca0 = decomposition.PCA(n_components=2).fit(k_data_scaled)
print(mapper_pca0.explained_variance_ratio_)
k_data_2d_pca = mapper_pca0.fit_transform(k_data_scaled)

idx_diff = np.array([i-j for i in range(10) for j in range(10)])
plt.scatter(k_data_2d_pca[:,0],k_data_2d_pca[:,1], s=5, 
            cmap=plt.cm.tab20, c=idx_diff)
# idx_diff.view(10,10)

#________________________________
# Plot the controllers on each diagonal in terms of error and effort

cmap = plt.cm.tab20#((np.arange(len(set(idx_diff)))))
# cmap[:,-1] = 1.

fig,ax = plt.subplots(1)
for i, diff in enumerate(set(idx_diff)):
    ax.plot(-k_labels[idx_diff==diff,0], -k_labels[idx_diff==diff,1],
            label=str(diff), color=cmap(i), ls=':')
ax.set_xlabel('error')
ax.set_ylabel('effort')
ax.legend()


# Chose the diagonal for the parametrization of the controller
data_scaler = preprocessing.StandardScaler().fit(k_data[idx_diff==-3,:])
k_data_scaled = data_scaler.transform(k_data[idx_diff==-3,:])

chosen_conts_scaled = k_data_scaled[idx_diff==-3,:]
cont_labels = k_labels[idx_diff==-3,:]
plt.scatter(-k_labels[:,0], -k_labels[:,1], s=5)
plt.xlabel('error')
plt.ylabel('effort')

mapper_pca = decomposition.PCA(n_components=1).fit(chosen_conts_scaled)
print(mapper_pca.explained_variance_ratio_)
cont_param = mapper_pca.fit_transform(chosen_conts_scaled).squeeze()
cont_param_str = ['%.2f'%item for item in cont_param]

fig,ax = plt.subplots(1)
ax.scatter(-cont_labels[:,0], -cont_labels[:,1], s=5)
ax.set_xlabel('error')
ax.set_ylabel('effort')
for i in range(len(cont_param)):
    ax.annotate('K='+cont_param_str[i], (-cont_labels[i,0], -cont_labels[i,1]))
    
plt.plot(cont_param, -10-np.log2(-cont_labels[:,0]), label='log2(error)')
plt.plot(cont_param, 10*cont_labels[:,1], label='effort')
plt.xlabel('K')
plt.legend()
plt.yticklabels()

# 
# Select the diagonal band of the controller table that starts with i=0,j=3.
# Project the controller vector onto a 1D space => K_star. 
# Make sure that error and effort change linearly as a function of K_star.
