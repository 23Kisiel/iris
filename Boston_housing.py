import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as geek


from sklearn.linear_model import LinearRegression

plt.style.use('ggplot')


boston = pd.read_csv('Boston.csv')
"""
         DESCRIBE:
print(boston.heads())
CRIM - crim - per capita crime rate
NX   - nox  - nitric oxides concentration
RM   - rm   - average number of rooms per dwelling
MEDV - medv - median value of owner occupied homes (in thousands $) 
"""

X = boston.drop('medv', axis = 1).values
y = boston['medv'].values


### task 1 - predict the price from a single feature (average number of rooms in block)
X_rooms = X[:,5]
# print(type(X_rooms), type(y)) <class 'numpy.ndarray'> <class 'numpy.ndarray'>
y = y.reshape(-1,1)
X_rooms = X_rooms.reshape(-1,1)

# plotting house values vs. number of rooms
'''
plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()
'''
#fitting regression model
reg = LinearRegression()
reg.fit(X_rooms, y)
prediction_space = geek.linespace(min(X_rooms),max(X_rooms)).reshape(-1,1)