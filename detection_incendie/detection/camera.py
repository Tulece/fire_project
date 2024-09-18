# detection_incendie/detection/camera.py

import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('detection_incendie/detection/model/fire_detection_model.h5')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret, frame = self.video.read()
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        if prediction[0][0] > 0.5:
            label = 'Fire Detected'
            color = (0, 0, 255)
        else:
            label = 'No Fire'
            color = (0, 255, 0)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
