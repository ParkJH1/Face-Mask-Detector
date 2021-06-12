# 07. keras_load_model.py
import tensorflow as tf
import matplotlib.pyplot as plt

# 모델 로드
model = tf.keras.models.load_model('../models/mymodel')

print(model.summary())
