# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 21:42:05 2016

@author: matt
"""

import itertools
import numpy as np 
import pandas as pd
from pandas import DataFrame
import statsmodels.formula.api as sm
import statsmodels.api as sma
import sklearn as sk 
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn

data = pd.read_csv('mortality.csv')
data['SO2'] = data['SO@']
data = data[['PREC', 'EDUC', 'NONW', 'SO2', 'MORT']]

# Correlations
pd.scatter_matrix(data)
data.corr().round(2)
