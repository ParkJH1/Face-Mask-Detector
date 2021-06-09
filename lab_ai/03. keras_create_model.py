# 03. keras_create_model.py
import tensorflow as tf

model = tf.keras.applications.ResNet101()
print(model.summary())
