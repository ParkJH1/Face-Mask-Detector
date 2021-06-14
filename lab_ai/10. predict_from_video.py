from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# https://drive.google.com/file/d/1GpZWS7339vymqQwvmV2kZ4h5TgzT0Q7Z/view?usp=sharing
#
# https://drive.google.com/file/d/1zypxcMVbZE_KzTf5vbDQobbllZRgSwKs/view?usp=sharing
#
# https://drive.google.com/file/d/1MGkYVIr1vTLJr4YcV3tZ4bojP-PWYCll/view?usp=sharing
#


resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

face_mask_recognition_model = cv2.dnn.readNet('../models/face_mask_recognition.prototxt', '../models/face_mask_recognition.caffemodel')
mask_detector_model = tf.keras.models.load_model('../models/mymodel')

cap = cv2.VideoCapture('../data/04.mp4')

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    face_mask_recognition_model.setInput(blob)
    dets = face_mask_recognition_model.forward()

    result_image = image.copy()

    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue

        left = int(dets[0, 0, i, 3] * w)
        top = int(dets[0, 0, i, 4] * h)
        right = int(dets[0, 0, i, 5] * w)
        bottom = int(dets[0, 0, i, 6] * h)

        face = image[top:bottom, left:right]

        face_image = image[top:bottom, left:right]
        face_image = cv2.resize(face_image, dsize=(224, 224))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        rc_face_image = resize_and_crop(np.array([face_image]))

        predict = mask_detector_model.predict(rc_face_image)
        if predict[0][0] > 0.5:
            color = (0, 0, 255)
            label = 'without_mask'
        else:
            color = (0, 255, 0)
            label = 'with_mask'

        cv2.rectangle(result_image, pt1=(left, top), pt2=(right, bottom), thickness=2, color=color,
                      lineType=cv2.LINE_AA)
        cv2.putText(result_image, text=label, org=(left, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)

        # face_input = cv2.resize(face, dsize=(224, 224))
        # face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        # face_input = preprocess_input(face_input)
        # face_input = np.expand_dims(face_input, axis=0)

        # mask, nomask = model.predict(face_input).squeeze()
        #
        # if mask > nomask:
        #     color = (0, 255, 0)
        #     label = 'Mask %d%%' % (mask * 100)
        # else:
        #     color = (0, 0, 255)
        #     label = 'No Mask %d%%' % (nomask * 100)
        #
        # cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        # cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
        #             color=color, thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('result', result_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
