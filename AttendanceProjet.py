from pydoc import classname

import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
from Basic import faceDis

path= 'AttendanceImages'
images=[]
classnames =[]
myList = os.listdir(path)
#print(myList)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])
#print(classnames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(names):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtString}')


encodlistknow=findEncodings(images)
#print('encoding Complete')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS= cv2.resize(img, (0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facescurframe = face_recognition.face_locations(imgS)
    encodescurframe = face_recognition.face_encodings(imgS,facescurframe)
    for encodeFace,faceloc in zip(encodescurframe,facescurframe):
        matches = face_recognition.compare_faces(encodlistknow,encodeFace)
        faceDis = face_recognition.face_distance(encodlistknow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            print(name)
            y1,x1,y2,x2 = faceloc
            y1, x1, y2, x2 = y1*4,x1*4,y2*4,x2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

cv2.imshow('Wecam',img)
cv2.waitKey(1)
"""
imgElon=face_recognition.load_image_file("image_basic/elonm.jpg")
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file("image_basic/bilgat.jpg")
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)"""