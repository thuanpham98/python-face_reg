import numpy as np 
import cv2 
import pickle

face_cas=cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels={"thuan":0}
with open("pickle/labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}


cap=cv2.VideoCapture(0)

while(True):
    #capture video frame
    ret,frame =cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, scaleFactor =1.3,minNeighbors=3)
    for(x,y,w,h) in faces:
        # print(x,y,w,h)

        # save picture as matrix
        roi_gray =gray[y:y+h, x:x+w]
        roi_color =frame[y:y+h, x:x+w]

        #regonizer 
        id_,conf = recognizer.predict(roi_gray)
        # print(id_)
        print(conf)
        if conf>=30 :
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            strole =3 
            size=4
            cv2.putText(frame,name,(x,y),font,size,color,strole,cv2.LINE_AA)

        #import file picture
        img_item1 = "images/" + str(1) + ".png"
        img_item2 = "images/my2.png"
        cv2.imwrite(img_item1,roi_gray)
        cv2.imwrite(img_item2,roi_color)

        #draw rectangle around face 
        color = (255,0,0)
        strole =5 # line Border
        end_cord_x=x +w
        end_cord_y=y +h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,strole)

    #Display the result frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()