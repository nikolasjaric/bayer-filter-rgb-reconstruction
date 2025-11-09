import os
from glob import glob
import cv2
import numpy as np

def bayer_mosaic_generator(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    print(f"Generating Bayer mosaic from: {input_path} to: {output_path}")

    search_pattern = os.path.join(input_path, "*.png")
    dataset = glob(search_pattern)

    for image in dataset:
        image_cv2 = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if image_cv2 is None:
            raise FileNotFoundError(f"Error: Could not load image: {image}")

        # Imena datoteka
        base_name = os.path.basename(image)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(output_path, f"mosaic_{name}.png")

        # BGR -> RGB
        color_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        H, W, C = color_image.shape
        if C != 3:
            raise ValueError(f"Expected 3-channel RGB image, got {C} channels")

        # --- tvoje postojeće (GRBG) maske ---
        R_mask  = np.zeros((H, W), dtype=bool)  # R at even row, odd col
        G1_mask = np.zeros((H, W), dtype=bool)  # G at even row, even col
        G2_mask = np.zeros((H, W), dtype=bool)  # G at odd row,  odd col
        B_mask  = np.zeros((H, W), dtype=bool)  # B at odd row,  even col

        R_mask[0::2, 1::2] = True
        G1_mask[0::2, 0::2] = True
        G2_mask[1::2, 1::2] = True
        B_mask[1::2, 0::2] = True
        G_mask = G1_mask | G2_mask

        # 3-kanalni "packed" mozaik: vrijednosti samo na svojim CFA pozicijama, ostalo 0
        packed = np.zeros((H, W, 3), dtype=color_image.dtype)
        packed[..., 0][R_mask] = color_image[..., 0][R_mask]  # R
        packed[..., 1][G_mask] = color_image[..., 1][G_mask]  # G
        packed[..., 2][B_mask] = color_image[..., 2][B_mask]  # B

        # Direktno spremanje SIROVIH piksela (bez figure renderinga!)
        ok = cv2.imwrite(save_path, cv2.cvtColor(packed, cv2.COLOR_RGB2BGR))
        if not ok:
            raise IOError(f"Failed to write: {save_path}")
