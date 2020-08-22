# -*- coding: utf-8 -*-

#Gerekli Kütüphaneleri Ekliyoruz
import numpy as np #Liste ve array işlemleri için
import pandas as pd#Dataframeler için
import random#rastgele değer türetmek için


#Bizde soruda istenen rastgele atamaları yapalım
sumiktarı = []
for i in range(8):
    sumiktarı.append(random.randint(7,16))
gübre_tekrar = []
for i in range(8):
    gübre_tekrar.append(random.randint(1,2))
budanma = []
for i in range(8):
    budanma.append(random.randint(0,1))
toplanma = []
for i in range(8):
    toplanma.append(random.randint(0,1))
toprak = []
for i in range(8):
    toprak.append(random.randint(0,1))
toprakdurum = []
for i in range(8):
    toprakdurum.append(random.randint(0,1))
tarlacevresi = []
for i in range(8):
    tarlacevresi.append(random.randint(0,1))
verim = []
for i in range(8):
    verim.append(float(random.uniform(0.7,0.9)))


index=["1.ağac","2.ağac","3.ağac","4.ağac","5.ağac","6.ağac","7.ağac","8.ağac"]#Kullanmak istersek diye
#Altta verileri sözlüğe çevirdik dataframe e kolaylıkla dönüştürebilmek için
datas = {"sumiktarı":sumiktarı,"gübre_tekrar":gübre_tekrar,"budanma":budanma,"toplanma":toplanma,"toprak":toprak,"toprakdurum":toprakdurum,"tarlacevresi":tarlacevresi,"verim":verim}
#dataframe e çevirdik
df = pd.DataFrame(datas)                             

X = df.iloc[:,:7].values#Özellikleri belirttik
y = df.iloc[:,7].values#Sonucları belirttik

#Yapay Sinir Ağı işlemleri için keras ı ekledik
import keras 
from keras.models import Sequential
from keras.layers import Dense

predictier = Sequential()#Katmanı başlattık
#10 processli bir hidden layer ekledik
predictier.add(Dense(output_dim=10,init="uniform",activation="relu",input_dim=7))
#Çıktı layerını ayarladık
predictier.add(Dense(output_dim = 1,init = "uniform",activation="sigmoid"))

#Sinir ağımızı derledik
predictier.compile(optimizer = "adam",loss = "binary_crossentropy",metrics=["accuracy"])

#Sinir ağını verilerimizle eğitelim
predictier.fit(X,y,batch_size=32,epochs=100)

#Tahmin yapalım
print("Örnek Tahmin : "+str(predictier.predict(np.array([13,2,0,1,0,0,0]).reshape(1,-1))))









    
    
    
    
    
    
    
    
    
    
    
    
