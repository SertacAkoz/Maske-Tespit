import cv2 
import pickle 
import numpy as np 
import time

classIndex = None

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.equalizeHist(img)
#    img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    img = img/255
    return img 

pickle_in = open("mask_model.p","rb")
model = pickle.load(pickle_in) 
#sınıflandırıcı
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

predictList = ["Maske Var", "Maske Yok"]

cap = cv2.VideoCapture(0)
crop_image = None
while True:
    ret, frame = cap.read()

    if ret:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
        color = (255,0,0)
        if classIndex == 0:
            color = (0,255,0)
        elif classIndex == 1:
            color = (0,0,255)
        for (x,y,w,h) in face_rect:
           cv2.rectangle(frame, (x,y), (x+w,y+h),color,2)
           #if x is not None:
           crop_image = frame[ y:y+h,  x:x+w]
        
        cv2.imshow("face detect", frame)
        #time.sleep(0.5)
        if crop_image is not None:
            try:
                cv2.imshow("crop", crop_image)

                crop_image = preProcess(crop_image)

                crop_image = cv2.resize(crop_image, (224, 224)) # input shape
                
                crop_image = crop_image.reshape(1,224, 224,3) # 1 adet resim, (224, 224) boytunda, channel = 1
                
                classIndex = int(model.predict_classes(crop_image))

                predictions = model.predict(crop_image)
                probVal = np.amax(predictions)

                if probVal > 0.8:
                    cv2.putText(frame, str(predictList[classIndex])+"     " + str(probVal),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,color,1,cv2.LINE_AA)
                    print(predictList[classIndex],probVal)
            except:
                print("Yuz Tespit Edilemedi")
    
    if cv2.waitKey(1) & 0xFF == ord("q"):break 

cap.release()
cv2.destroyAllWindows()
