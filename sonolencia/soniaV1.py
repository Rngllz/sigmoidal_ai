# CÃ³digo baseado no artigo do Adrian Rosebrock
# https://bit.ly/3ht4dbG

# importar as bibliotecas
from scipy.spatial import distance as dist
import cv2
import dlib
import numpy as np
import time
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import matplotlib.pyplot as plt

# dlib detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
vs = VideoStream(src=1).start()
time.sleep(2.0)


# desenhar um objeto do tipo figure
y = [None] * 100
x = np.arange(0,100)
fig = plt.figure()
ax = fig.add_subplot(111)
li, = ax.plot(x, y)



(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# video processing pipeline
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # extrair coordenadas dos olhos e calcular a proporÃ§Ã£o de abertura
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # ratio mÃ©dia para os dois olhos
        ear = (leftEAR + rightEAR) / 2.0
        
        print(ear)
        
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 250, 0), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    
# clean
cv2.destroyAllWindows()
vs.stop()
