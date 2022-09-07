#!/usr/bin/env python
'''
This script contains code for applying locally weighted linear regression model (trained using first 20 observations of
training data) to test data and estimating MSE
Author: Sai Venkata Krishnaveni Devarakonda
Date: 03/19/2022
'''

import numpy as np
import matplotlib.pyplot as plt
from utilities import Load_data_2,get_weights,get_coef_local_lin_reg,mse

#Load data
str_1c_testdata_path = './1c_test_data.txt'
str_1b_traindata_path = './1b_training_data.txt'
arr2D_test_feature_set, arr1D_test_targets, arr1D_test_x = Load_data_2(str_1c_testdata_path)
arr2D_train_feature_set, arr1D_train_targets, arr1D_train_x = Load_data_2(str_1b_traindata_path)
arr2D_train_feature_set, arr1D_train_targets, arr1D_train_x = arr2D_train_feature_set[:21,:], arr1D_train_targets[:21,], arr1D_train_x[:21,]
gamma =0.1


#predicting targets for test features
pred_test_targets = []
for i in range(len(arr1D_test_x)):
    test_feature = arr1D_test_x[i]
    # calculating weights
    weights = get_weights(arr1D_train_x, test_feature, gamma)
    coef = get_coef_local_lin_reg(weights,arr2D_train_feature_set, arr1D_train_targets)
    #predicting target for test feature
    pred_y = np.dot(coef, arr2D_test_feature_set[i])
    pred_test_targets.append(pred_y)
pred_test_targets = np.asarray(pred_test_targets)

#predicting targets for train features
pred_train_targets = []
for i in range(len(arr1D_train_x)):
    test_feature = arr1D_train_x[i]
    # calculating weights
    weights = get_weights(arr1D_train_x, test_feature, gamma)
    coef = get_coef_local_lin_reg(weights,arr2D_train_feature_set, arr1D_train_targets)
    #predicting target for test feature
    pred_y = np.dot(coef, arr2D_train_feature_set[i])
    pred_train_targets.append(pred_y)
pred_train_targets = np.asarray(pred_train_targets)

#Calculating mean squared error for train data and test data
mse_test = mse(pred_test_targets, arr1D_test_targets)
mse_train = mse(pred_train_targets, arr1D_train_targets)
print('LocalLinReg train MSE=' + str(mse_train))
print('LocalLinReg test MSE =' + str(mse_test))

#Plot locallinreg resulting function together with data points
# sort x - get indices
x_sorted = np.argsort(arr1D_test_x)
plt.plot(arr1D_test_x[x_sorted], arr1D_test_targets[x_sorted], '*g')
# plotting given features with predicted targets
plt.plot(arr1D_test_x[x_sorted], pred_test_targets[x_sorted], '-b')
plt.xlabel('test x')
plt.ylabel('signal amplitude(y)')
plt.title('Locally wt linreg(20) for test data d=0')
plt.legend(['original data points','locally wt linreg'])
plt.show()