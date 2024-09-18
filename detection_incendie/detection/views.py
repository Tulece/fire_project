from django.shortcuts import render
import tensorflow as tf
import numpy as np
import cv2

# Charger le modèle
model = tf.keras.models.load_model('detection_incendie/detection/model/fire_detection_model.h5')

def home(request):
    return render(request, 'detection/home.html')

def detect_fire(request):
    return render(request, 'detection/detect_fire.html')

from django.http import StreamingHttpResponse
from .camera import VideoCamera

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
