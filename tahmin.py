import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import matplotlib.pyplot as plt

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
model=load_model('kerasileuygulama')




def resmiklasordenal(dosyaadi):
    resim = cv2.imread("%s" %dosyaadi) #Opencv den resmi çektik.dönüş larak resmi dönecek.
    return resim


#Tahmin edebilmek için for i in range komutu ile veriseti klasöründeki test için ayrılan resimlerin indis aralığı verilip tahmin metodu ile kedi veya çiçekmi olduğu tahmin edildi.

x=range(30,80)
for i in x:
        
       
        #Herseferinde klasordenalinanresim i sıfırlasın.
        klasordenalinmisresim=0  
        girisverisi_i= np.array([])
        #resimleri alacağım yolu bir değişkende tutarım.
        string='veriseti/%s.jpg'%i
        print(string)
        klasordenalinmisresim = resmiklasordenal(string)
        #Klasörden alınan resimin boyutu ayarlanır.
        boyutlandirilmisresim = cv2.resize(klasordenalinmisresim,(224,224))
        #Boyutlandirilmis resmi girisverisi matrisine almam lazım.
        #boyutlandirilmisresim'i girisverisi matrisine at.
        girisverisi_i= np.append(girisverisi_i,boyutlandirilmisresim)
        #reshape komutu ile dizi kaç boyutlu ise onu verir
        girisverisi_i=np.reshape(girisverisi_i,(-1,224,224,3))
        #Sonra dosyadan okunup alınan resimler bizim girisverimiz adlı datasetimize kaydettik.Bu sayede her seferinde gidip dosyadan okuma derdinden kurtulduk.
        ag_cikisi = model.predict([girisverisi_i])[0]
        i=i+1
        if np.argmax(ag_cikisi) == 0:
     
          print('Çiçek')
        else:
       
         print('Kedi')
   
      









