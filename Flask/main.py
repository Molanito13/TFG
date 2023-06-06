import cv2
import time
import numpy
import tensorflow as tf
import skimage
import os
import base64
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from deploy import *

#frame = numpy.array(skimage.io.imread('./archive_grande/ASL_Dataset/Train/H/5.jpg'))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la fuente de video")
    exit()

while True:
    # Get image frame
    success, img = cap.read()

    cv2.imshow("Image", receive_image(img))

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
