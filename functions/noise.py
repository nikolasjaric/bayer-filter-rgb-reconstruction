import numpy as np
from pathlib import Path
import cv2
import os

def add_noise(original_folder, noise_folder):
    originals = Path(original_folder).glob('*.png')
    sigma = 20
    mean = 0
    for original in originals:
        img = cv2.imread(original)
        gauss = np.random.normal(mean, sigma, img.shape)
        gauss = gauss.reshape(img.shape)
        noised = img + gauss

        base_name = os.path.basename(original)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(noise_folder, f"noise_{name}.png")
        cv2.imwrite(save_path, noised)
    return