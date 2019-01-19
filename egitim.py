import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

#Klasörden datasetimize çektiğimiz resimleri yükleriz.
girisverisi = np.load("girisverimiz.npy")

#Çıkışverisi etiketleme işlemi yapılıyor.
cikisverisi = np.array([ [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]      ])
#Dilerseniz bu şekilde içinde kaç veri var düz olarak görebilirsiniz.Sonuı. olarak(30,2) döner.2 li şekilde 30 tane matris var demek.--> print(cikisverisi.shape)



#Bu kısım validation split için
splitverisi = girisverisi[1:4]  #Giriş verisini ilk 3 elemanını alsın
splitverisi = np.append(splitverisi,girisverisi[24:27]) #Split verisini tekrar giriş verisini son 3 elemaınıa ekledi
splitverisi= splitverisi.reshape(-1,224,224,3)
splitcikis = np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]])




model = Sequential()
model.add(Conv2D(50,11,strides=(4,4),input_shape=(224,224,3)))#50 tane özellik haritam,11 tane kernel size var
model.add(MaxPooling2D(5,5))
model.add(Conv2D(50,5)) #50 tane özellik haritam .5 tane de kernel size var.
model.add(Conv2D(50,3))
model.add(Conv2D(50,3))
model.add(Conv2D(50,2))
model.add(Flatten())    #Tüm ağırlık matrisleri düm düz hale gelir.

model.add(Dense(4096,activation='relu')) #Full-connected katmanı ile tüm veriler birbirine bağlanır.
model.add(Dropout(0.3)) #Overfitting i önlemek için

model.add(Dense(4096,activation='relu')) #Full-connected katmanı ile tüm veriler birbirine bağlanır.
model.add(Dense(2))  #Enson katmandan sonra 2 veriden birini getirecek.Ya kedi ya çiçek.
model.add(Activation('softmax')) #En son katmanım aktivasyon katmanıdır.

model.compile(loss='categorical_crossentropy',optimizer= 'adam', metrics=['accuracy'])
model.summary() #Ağı özetler
model.fit(girisverisi,cikisverisi,batch_size=2,epochs=7,validation_data=(splitverisi,splitcikis))  #Ağıma girişverisi ,çıkış verisi lazım.Batch_size:AYNI anda kaç veri yükleyip tranin etsin.Epochs=tekrar sayısı.