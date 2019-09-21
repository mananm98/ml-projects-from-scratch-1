import numpy as np
import os
import cv2
class_id = 0
def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())


def knn(train,test,k = 5):
    d = []
    m = train.shape[0]
    for i in range(m):
        dist = distance(test,train[i,:-1])
        d.append([dist,train[i, -1]])

    d = np.array(sorted(d,key = lambda x:x[0]))[:,-1]
    d = d[:k]
    t = np.unique(d,return_counts= True)
    idx = np.argmax(t[1])
    pred = int(t[0][idx])

    return pred





cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("/Users/mananmehta/Desktop/p/machine-learning-june-2019/Lecture-03,04 FaceRecognition Linear Regression/haarcascade_frontalface_alt.xml")
data_path = "./Facedata/"
data = []
dict = {}
label = []


for fname in os.listdir(data_path):
    if fname.endswith(".npy"):

        dict[class_id] = fname[:-4]
        file = np.load(data_path+fname)

        for i in range(file.shape[0]):
            data.append(file[i])
        for i in range(file.shape[0]):
            label.append(class_id)

        class_id += 1


data = np.array(data)
print(data.shape)
label = np.array(label)
print(label.shape)


trainset = np.concatenate((data,label[:,None]), axis = 1)
print(trainset.shape)


# getting test data
while True:
    ret,frame = cam.read()

    key_pressed  = cv2.waitKey(1)&0xFF
    if key_pressed == ord('q'):
        break

    frame = cv2.resize(frame,(600,400))

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    if(len(faces) == 0):
        cv2.imshow("video",frame)
        continue


    for face in faces:
        x,y,w,h = face
        detected_face = frame[y-10:y+h+10,x-10:x+w+10]
        detected_face = cv2.resize(detected_face,(100,100))
        detected_face = np.array(detected_face).reshape((1,-1))
        prediction = knn(trainset,detected_face)
        text = dict[prediction]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,text,(x,y), font, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,x+h),(0,255,255),2)

    cv2.imshow("video",frame)

cam.release()
cv2.destroyAllWindows()
