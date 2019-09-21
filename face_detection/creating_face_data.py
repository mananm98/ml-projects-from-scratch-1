import cv2
import numpy as np
# read a video stream and display it
cnt = 0
#camera object
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("/Users/mananmehta/Desktop/p/machine-learning-june-2019/Lecture-03 FaceRecognition/haarcascade_frontalface_alt.xml")
face_data = []
user_name = input("enter your name")

while True:
    ret,frame = camera.read()
    frame = cv2.resize(frame,(600,400))

    if ret == False:
        continue

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break



    faces = face_cascade.detectMultiScale(frame,1.3,5)
    #print(faces)
    if(len(faces)==0):
        cv2.imshow("Video",frame)
        continue

    for face in faces:
        x,y,w,h = face

        face_section = frame[y-10:y+h+10,x-10:x+w+10]
        face_section = cv2.resize(face_section,(100,100))
        cv2.rectangle(frame,(x,y),(x+w,x+h),(0,255,255),2)
        if cnt%5 == 0:
            print("taking picture ",int(cnt/10))
            face_data.append(face_section)
        cnt += 1


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video",frame)
    cv2.imshow("video gray",face_section)

#save data in numpy file
print("Total faces",len(face_data))
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))

np.save("FaceData/"+user_name+".npy",face_data)
print("Saved at FaceData/"+user_name+".npy")
print(face_data.shape)


camera.release()
# to close the window that shows our output
cv2.destroyAllWindows()
