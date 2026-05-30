import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

clean_images = []
noisy_images = []

clean_folder = "data/processed/clean"
noisy_folder = "data/processed/noisy"

files = os.listdir(clean_folder)

for file in files:

    clean = cv2.imread(
        os.path.join(clean_folder, file),
        cv2.IMREAD_GRAYSCALE
    )

    noisy = cv2.imread(
        os.path.join(noisy_folder, file),
        cv2.IMREAD_GRAYSCALE
    )

    clean = clean / 255.0
    noisy = noisy / 255.0

    clean_images.append(clean)
    noisy_images.append(noisy)

clean_images = np.array(clean_images)
noisy_images = np.array(noisy_images)

clean_images = clean_images.reshape(
    -1, 128, 128, 1
)

noisy_images = noisy_images.reshape(
    -1, 128, 128, 1
)

print("Dataset Shape:", clean_images.shape)

# CNN Autoencoder

input_img = Input(shape=(128,128,1))

x = Conv2D(
    32,
    (3,3),
    activation="relu",
    padding="same"
)(input_img)

x = MaxPooling2D((2,2))(x)

x = Conv2D(
    64,
    (3,3),
    activation="relu",
    padding="same"
)(x)

encoded = MaxPooling2D((2,2))(x)

x = Conv2D(
    64,
    (3,3),
    activation="relu",
    padding="same"
)(encoded)

x = UpSampling2D((2,2))(x)

x = Conv2D(
    32,
    (3,3),
    activation="relu",
    padding="same"
)(x)

x = UpSampling2D((2,2))(x)

decoded = Conv2D(
    1,
    (3,3),
    activation="sigmoid",
    padding="same"
)(x)

model = Model(input_img, decoded)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

history = model.fit(
    noisy_images,
    clean_images,
    epochs=10,
    batch_size=16,
    validation_split=0.1
)

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/cnn_medical_enhancer.keras"
)

print("Model saved successfully!")