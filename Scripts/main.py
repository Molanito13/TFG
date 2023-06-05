#from cvzone.FaceDetectionModule import FaceDetector
#from cvzone.HandTrackingModule import HandDetector
import cv2
import time
import numpy
import tensorflow as tf
import skimage
import os
letras = numpy.array(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","Nothing","O","P","Q","R","S","Space","T","U","V","W","X","Y","Z"])
os.chdir("..")
#frame = numpy.array(skimage.io.imread('./archive_grande/ASL_Dataset/Train/H/5.jpg'))
modelo = tf.keras.models.load_model('./Modelos/ModeloTrain.keras')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la fuente de video")
    exit()

currentTime = 0
previousTime = 0

while True:
    # Get image frame
    success, img = cap.read()

    print(type(img))

    img2 = cv2.resize(img, (70,70))

    p = modelo.predict(numpy.array([img2]))
    
    print(f"La clase predicha es {letras[numpy.argmax(p)]} con una probabilidad de {numpy.max(p)*100}%")

    # Calculate FPS and display it
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, f"FPS: {letras[numpy.argmax(p)]}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
    # Display the image
    cv2.imshow("Image", img)
    print(img.shape)

    

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
