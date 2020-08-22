# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:24:36 2020

@author: Murat Çiçek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veri = pd.read_csv('C:\\Users\\Murat\\Desktop\\ysa\\data.csv')

X = veri["Days"].values.reshape(-1,1)
y = veri["Prices"].values.reshape(-1,1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

from sklearn.preprocessing import Normalizer
Sc = Normalizer()
X_train = Sc.fit_transform(X_train)
X_test = Sc.fit_transform(X_test)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

x_poly = poly.fit_transform(X_train)

sample = y[:40]

lr.fit(x_poly,sample)

plt.title("Günlere Göre Polynomial Alış Tahmini")
plt.scatter(X,y,color = "red", label = "Gerçek Değer")
plt.plot(X,lr.predict(poly.fit_transform(X)), color = "green", label = "Tahmin")
plt.legend()
plt.show()

print("11.gün altın alış fiyatı : ")
print(lr.predict(poly.fit_transform(np.array([11]).reshape(-1,1))))
