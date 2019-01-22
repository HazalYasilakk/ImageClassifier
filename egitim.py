import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization

#Klasörden datasetimize çektiğimiz resimleri yükleriz.
girisverisi = np.load("girisverimiz.npy")
girisverisi = np.reshape(girisverisi,(-1,224,224,3))
#Çıkış verisi etiketleme işlemi yapılıyor.
cikisverisi = np.array([ [1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]      ])
#Dilerseniz bu şekilde içinde kaç veri var düz olarak görebilirsiniz.Sonuı. olarak(30,2) döner.2 li şekilde 30 tane matris var demek.--> print(cikisverisi.shape)



#Bu kısım validation split için
splitverisi = girisverisi[1:6]  #Giriş verisini ilk 6 elemanını alsın
splitverisi = np.append(splitverisi,girisverisi[24:29]) #Split verisini tekrar giriş verisini son elemaınıa ekledi
splitverisi= splitverisi.reshape(-1,224,224,3)
splitcikis = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1]])




model = Sequential()


model.add(Conv2D(50,11,strides=(4,4),input_shape=(224,224,3)))#50 tane özellik haritam,11 tane kernel size var
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,3))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(MaxPooling2D(5,5))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(MaxPooling2D(3,3))
model.add(Conv2D(50,2))


model.add(Flatten())    #Tüm ağırlık matrisleri düm düz hale gelir.

model.add(Dense(1000,activation='relu')) #Full-connected katmanı ile tüm veriler birbirine bağlanır.
model.add(Dropout(0.3)) #Overfitting i önlemek için

model.add(Dense(1000,activation='relu')) #Full-connected katmanı ile tüm veriler birbirine bağlanır.
model.add(Dropout(0.3)) #Overfitting i önlemek için
model.add(Dense(2))  #Enson katmandan sonra 2 veriden birini getirecek.Ya kedi ya çiçek.
model.add(Activation('softmax')) #En son katmanım aktivasyon katmanıdır.

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.00001), metrics=['accuracy'])
model.summary() #Ağı özetler
print(splitverisi.shape)
model.fit(girisverisi/255,cikisverisi,batch_size=1,epochs=40,validation_data=(splitverisi,splitcikis))  #Ağıma girişverisi ,çıkış verisi lazım.Batch_size:AYNI anda kaç veri yükleyip tranin etsin.Epochs=tekrar sayısı.
model.save("kerasileuygulama")
