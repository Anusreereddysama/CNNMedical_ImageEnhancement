import cv2
import os
import numpy as np
from skimage.util import random_noise

os.makedirs("data/processed/clean", exist_ok=True)
os.makedirs("data/processed/noisy", exist_ok=True)

folders = ["yes", "no"]

count = 0

for folder in folders:

    folder_path = os.path.join(
        "data/raw",
        folder
    )

    for file in os.listdir(folder_path):

        image_path = os.path.join(
            folder_path,
            file
        )

        img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        img = cv2.resize(
            img,
            (128,128)
        )

        clean_name = f"{folder}_{file}"

        cv2.imwrite(
            f"data/processed/clean/{clean_name}",
            img
        )

        noisy = random_noise(
            img,
            mode="gaussian",
            var=0.01
        )

        noisy = (
            noisy * 255
        ).astype(np.uint8)

        cv2.imwrite(
            f"data/processed/noisy/{clean_name}",
            noisy
        )

        count += 1

print(f"{count} images processed successfully!")