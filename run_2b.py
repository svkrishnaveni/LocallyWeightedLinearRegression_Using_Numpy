#!/usr/bin/env python
'''
This script contains code for applying locally weighted linear regression model trained using training data and plotting
resulting function with original datapoints
Author: Sai Venkata Krishnaveni Devarakonda
Date: 03/19/2022
'''

from utilities import plot_2b
# load the traindata generated from question 1b for function depth d=0

str_path_2b_traindata='./1b_training_data.txt'
plot_2b(str_path_2b_traindata=str_path_2b_traindata, save_figures=True)


