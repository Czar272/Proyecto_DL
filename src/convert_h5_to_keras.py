import json, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

H5_PATH = "models/mobilenetv2.h5"
LABELS_PATH = "labels.json"
KERAS_OUT = "models/mobilenetv2.keras"
IMG_SIZE = 224

class_names = json.load(open(LABELS_PATH, "r", encoding="utf-8"))
num_classes = len(class_names)

base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.load_weights(H5_PATH, by_name=True, skip_mismatch=True)

model.save(KERAS_OUT)
print(f"Guardado: {KERAS_OUT}")
