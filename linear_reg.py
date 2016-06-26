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

data = pd.read_csv('/home/matt/example_code/mortality.csv')
data['SO2'] = data['SO@']
data = data[['PREC', 'EDUC', 'NONW', 'SO2', 'MORT']]

# Correlations
pd.scatter_matrix(data)
data.corr().round(2)

# First order model
fm1 = sm.ols(formula = 'MORT ~ PREC + EDUC + NONW', data = data)
res_fm1 = fm1.fit()
res_fm1.summary()

# Interaction Term
fm1b = sm.ols(formula = 'MORT ~ PREC * EDUC * NONW', data = data)
res_fm1b = fm1b.fit()
res_fm1b.summary()

# Second order Model
formula =  'MORT ~ PREC + np.power(PREC,2) + EDUC + '
formula += 'np.power(EDUC,2) + NONW + np.power(PREC,2)'
fm1c = sm.ols(formula = formula, data = data)
res_fm1c = fm1c.fit()
res_fm1c.summary()

xnew = pd.DataFrame({'PREC': np.arange(10,60,5), 
		'EDUC': 10.97333, 'NONW':11.87})
res_fm1c.predict(xnew)

# Example 2, checks for collinearity
file =  "http://www2.isye.gatech.edu/~ymei"
file += "/7406/Handouts/prostate.csv"
prostate = pd.read_csv(file)

pd.scatter_matrix(prostate)

# Creates a matrix of box plots
for i, column in enumerate(prostate.columns):
	if i == 9:
		break
	plt.subplot(3,3,i+1)
	prostate.boxplot(column = column)

training = prostate.loc[prostate.train == 'T']
test = prostate.loc[prostate.train == 'F']

all_var = [col for col in prostate.columns if col != 'lspa']
lhs = '+'.join(all_var)
formula = 'lpsa ~ ' + lhs
model0 = sm.ols(formula = formula, data = training)









