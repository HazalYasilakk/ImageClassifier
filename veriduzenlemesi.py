import cv2
import numpy as np

#Resimleri dosyadan almam için fonksiyon tanımladdım.

def resmiklasordenal(dosyaadi):
    resim = cv2.imread(dosyaadi) #Opencv den resmi çektik.dönüş larak resmi dönecek.
    return resim
    
    #Benim veriseti adlı dosyamda 1-30 kadar resmim var ve hepsini almam gerek
    #Tüm resimleri alacağım bir matris yazarım
    
girisverisi = np.array([])
    
for i in range(30):
        #Herseferinde klasordenalinanresim i sıfırlasın.
        klasordenalinmisresim=0  
        i=i+1
        #resimleri alacağım yolu bir değişkende tutarım.
        string='veriseti/%s.jpg'%i
        klasordenalinmisresim = resmiklasordenal(string)
        #Klasörden alınan resimin boyutu ayarlanır.
        boyutlandirilmisresim = cv2.resize(klasordenalinmisresim,(224,224))
        #Boyutlandirilmis resmi girisverisi matrisine almam lazım.
        #boyutlandirilmisresim'i girisverisi matrisine at.
        girisverisi= np.append(girisverisi,boyutlandirilmisresim)
        #reshape komutu ile dizi kaç boyutlu ise onu verir
        girisverisi=np.reshape(girisverisi,(-1,224,224,3))
        #Sonra dosyadan okunup alınan resimler bizim girisverimiz adlı datasetimize kaydettik.Bu sayede her seferinde gidip dosyadan okuma derdinden kurtulduk.
        np.save("girisverimiz",girisverisi)

        print(girisverisi.shape)