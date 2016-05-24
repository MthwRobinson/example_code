import numpy as np 

import pandas as pd
from pandas import DataFrame
import statsmodels.formula.api as sm
import statsmodels.api as sma

import sklearn as sk 
from sklearn import linear_model
import matplotlib.pyplot as plt

# Data and Data Frames
x = np.arange(1,31)
x = np.arange(1,4,.5)

# 3 by 2 Matrix
y = np.array([1,2,3,1,2,3])
y = reshape(y,(3,2))

y[0,:] # select the first row
y[:,1] # select the second column
y[2,0] 

# column and row means
[mean(y[:,i]) for i in range(y.shape[1])] 
[mean(y[j,:]) for j in range(y.shape[0])]

# More examples
x = np.random.normal(size = 50)
y = np.random.normal(size = 50)
plt.scatter(x,y)

x = np.random.normal(size = 1000)
y = np.random.normal(size = 1000)
z = np.hstack((x,y+2))
plt.hist(z)

x = arange(1,20,.5)
w = 1 + x/2
y = x + w * np.random.normal(size = len(x))
df = DataFrame({'x':x,'y':y,'w':w})

## Linear regression
lm = linear_model.LinearRegression()
data = np.matrix(df)
x,y = data[:,1], data[:,2]
lm.fit(x,y)

lm = sm.ols(formula = 'y ~ x', data = df).fit()
print lm.summary()
exog = pd.DataFrame({'x':[10,15]})
lm.predict(exog)
lm.resid
lm.params

# Weighted Least Squares
nsamp = df.shape[0]
Y = np.array(df['y'])
X = np.c_[ np.ones(nsamp), np.array(df['x'])]
fm1 = sm.WLS(Y, X, weights = 1 / w**2)
res_fm1 = fm1.fit()
res_fm1.summary()

# Plots
plt.figure()
plt.scatter(x,y)
x_range = arange(0,20,.1)
exog = DataFrame({'x':x_range})
y_pred = lm.predict(exog)
plt.plot(x_range,y_pred)
plt.close()

res = lm.resid
fig = sma.qqplot(resid)
plt.show()

# KNN

file = "http://www2.isye.gatech.edu/~ymei/7406/Handouts/mixtureexample.csv"
df = pd.read_csv(file, header = True)
rows = [0,1,2,197,198,199]
df.ix[rows]

data = np.matrix(df)
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]
