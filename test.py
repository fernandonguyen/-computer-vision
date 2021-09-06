import cv2
import numpy as np
import matplotlib.pyplot as plt

def convertToRGB(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)

    return image_copy

#test_image = cv2.imread('Resources/faces.jpg')


#faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)
while True:
    success, img = cap.read()
    # cv2.imshow("Result", img)
    haar_cascade_face = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
    test_image_1 = detect_faces(haar_cascade_face, img)
    cv2.imshow("Result", test_image_1)
    # plt.show()
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()