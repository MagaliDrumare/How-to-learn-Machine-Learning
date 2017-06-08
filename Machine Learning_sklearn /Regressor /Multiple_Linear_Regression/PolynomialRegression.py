#Polynomial Regression 

import numpy as np 
import matplotlib.pyplot as pyplot
import pandas as pd 

dataset=np.read_csv('')
X=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,3].values 


#Fitting Linear Regression 
from sklearn.linear_model import LinearRegression 
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression 
from sklear.preprocessing import PolynomialFeatures 
poly_reg=PolynomialFeatures()
X_poly=poly_reg.fit.transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualising the Linear Regression Model 
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('Thruth or Bluff')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression Model 
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(X), color='blue')
plt.title('Thruth or Bluff')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
