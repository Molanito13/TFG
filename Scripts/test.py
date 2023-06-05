import numpy as np
#from cvzone.FaceDetectionModule import FaceDetector
#from cvzone.HandTrackingModule import HandDetector
import cv2
import time
import numpy
import tensorflow as tf
import skimage
import os
letras = numpy.array(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","Nothing","O","P","Q","R","S","Space","T","U","V","W","X","Y","Z"])

frame = numpy.array(skimage.io.imread('./archive_grande/ASL_Dataset/Train/H/5.jpg'))
modelo = tf.keras.models.load_model('./ModeloTrain3.keras')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la fuente de video")
    exit()

'''detector = HandDetector(detectionCon=0.5, maxHands=2)
faceDetector = FaceDetector()'''

currentTime = 0
previousTime = 0

while True:
    # Get image frame
    success, img = cap.read()

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # Definir el rango de colores para la piel blanca
    lower_skin = np.array([0,133,77],np.uint8)
    upper_skin = np.array([235,173,127],np.uint8)
    
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(img, img, mask=mask)

    '''# Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    # Find faces and draw bounding boxes
    faces, img = faceDetector.findFaces(img)'''

    img2 = cv2.resize(img, (70,70))

    '''img = numpy.array(img)'''

    '''if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)

            # Find Distance between two Landmarks. Could be same hand or different hands
            #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
            #length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw'''

    

    p = modelo.predict(numpy.array([img2]))
    print(f"La clase predicha es {letras[numpy.argmax(p)]} con una probabilidad de {numpy.max(p)*100}%")

    # Calculate FPS and display it
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(result, f"FPS: {letras[numpy.argmax(p)]}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
    # Display the image
    cv2.imshow("Image", result)

    

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
