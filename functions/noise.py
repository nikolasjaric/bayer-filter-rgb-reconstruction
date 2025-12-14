import numpy as np
from pathlib import Path
import cv2
import os

def add_noise(original_folder, noise_folder):
    originals = Path(original_folder).glob('*.png')
    for original in originals:
        img = cv2.imread(original)
        noise = np.random.normal(0, 50, img.shape)
        noised = img + noise
        noised = np.clip(noised, 0, 255).astype(np.uint8)

        base_name = os.path.basename(original)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(noise_folder, f"noise_{name}.png")
        cv2.imwrite(save_path, noised)
    return