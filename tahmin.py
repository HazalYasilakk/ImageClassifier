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
    



#output=model.predict(girisverisi)

#if output==[0,1]:
# print("Cat")
#else:
# print("Flower")  #1 0




girisverisi1= np.array([])
klasordenalinmisresim=0  
string='veriseti/test1.jpg'
klasordenalinmisresim = resmiklasordenal(string)
boyutlandirilmisresim = cv2.resize(klasordenalinmisresim,(224,224))
girisverisi1= np.append(girisverisi1,boyutlandirilmisresim)
girisverisi1=np.reshape(girisverisi1,(-1,224,224,3))
ag_cikisi = model.predict([girisverisi1])[0]
if np.argmax(ag_cikisi) == 0:
 print('Çiçek')
else:
 print('Kedi')




girisverisi2 = np.array([])
klasordenalinmisresim=0  
string='veriseti/test2.jpg'
klasordenalinmisresim = resmiklasordenal(string)
boyutlandirilmisresim = cv2.resize(klasordenalinmisresim,(224,224))
girisverisi2= np.append(girisverisi2,boyutlandirilmisresim)
girisverisi2=np.reshape(girisverisi2,(-1,224,224,3))
ag_cikisi = model.predict([girisverisi2])[0]
if np.argmax(ag_cikisi) == 0:
 print('Çiçek')
else:
 print('Kedi')


girisverisi3 = np.array([])
klasordenalinmisresim=0  
string='veriseti/kedi.jpg'
klasordenalinmisresim = resmiklasordenal(string)
boyutlandirilmisresim = cv2.resize(klasordenalinmisresim,(224,224))
girisverisi3= np.append(girisverisi3,boyutlandirilmisresim)
girisverisi3=np.reshape(girisverisi3,(-1,224,224,3))
ag_cikisi = model.predict([girisverisi3])[0]
if np.argmax(ag_cikisi) == 0:
 print('Çiçek')
else:
 print('Kedi')
 
 

girisverisi4 = np.array([])
klasordenalinmisresim=0  
string='veriseti/kedi1.jpg'
klasordenalinmisresim = resmiklasordenal(string)
boyutlandirilmisresim = cv2.resize(klasordenalinmisresim,(224,224))
girisverisi4= np.append(girisverisi4,boyutlandirilmisresim)
girisverisi4=np.reshape(girisverisi4,(-1,224,224,3))
ag_cikisi = model.predict([girisverisi4])[0]
if np.argmax(ag_cikisi) == 0:
 print('Çiçek')
else:
 print('Kedi')


girisverisi5 = np.array([])
klasordenalinmisresim=0  
string='veriseti/kedi2.jpg'
klasordenalinmisresim = resmiklasordenal(string)
boyutlandirilmisresim = cv2.resize(klasordenalinmisresim,(224,224))
girisverisi5= np.append(girisverisi5,boyutlandirilmisresim)
girisverisi5=np.reshape(girisverisi5,(-1,224,224,3))
ag_cikisi = model.predict([girisverisi5])[0]
if np.argmax(ag_cikisi) == 0:
 print('Çiçek')
else:
 print('Kedi')


girisverisi6 = np.array([])
klasordenalinmisresim=0  
string='veriseti/kedi3.jpg'
klasordenalinmisresim = resmiklasordenal(string)
boyutlandirilmisresim = cv2.resize(klasordenalinmisresim,(224,224))
girisverisi6= np.append(girisverisi6,boyutlandirilmisresim)
girisverisi6=np.reshape(girisverisi6,(-1,224,224,3))
ag_cikisi = model.predict([girisverisi6])[0]
if np.argmax(ag_cikisi) == 0:
 print('Çiçek')
else:
 print('Kedi')






#print(model.predict(girisverisi1)) #1 0








