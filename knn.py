import itertools

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

# Plotting by Group

file = "http://www2.isye.gatech.edu/~ymei/7406/Handouts/mixtureexample.csv"
df = pd.read_csv(file, header = False)
rows = [0,1,2,197,198,199]
df.ix[rows]

data = np.matrix(df)
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]

colors = itertools.cycle(['r','b'])
groups = df.groupby('y')
for name, group in groups:
	plt.scatter(group.x1,group.x2,
		color = next(colors), label = name)
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatterplot Example')

# KNN
from sklearn.neighbors import KNeighborsClassifier

px1 = np.arange(-2.6,4.3,0.1)
px1 = px1.reshape(len(px1),1)
px2 = np.arange(-2,2.95,0.05)
px2 = px2.reshape(len(px2),1)
px2new = np.array([px2[0] for j in range(len(px1))])
px2new = px2new.reshape(len(px2new),1)
xnew1 = np.hstack((px1,px2new))
for i in range(1,len(px2)):
	px2new = np.array([px2[i] for j in range(len(px1))])
	px2new = px2new.reshape(len(px2new),1)
	xnew1 = np.vstack((xnew1, np.hstack((px1,px2new))))

X = np.hstack((x1,x2))
neigh = KNeighborsClassifier(n_neighbors = 15)
neigh.fit(X, ravel(y))
pred = neigh.predict(xnew1)
pred_prob = neigh.predict_proba(xnew1)

x1cont = []; x2cont = []
prob_list = list(pred_prob[:,0])
for i in range(len(prob_list)):
	val = prob_list[i]
	if val > 0.45 and val < 0.55:
		x1cont.append(xnew1[i][0])
		x2cont.append(xnew1[i][1])
plt.plot(x1cont,x2cont)
