import os
import cv2
import numpy as np
from glob import glob
from scipy.ndimage import convolve


def bilinear_interpolation(input_path, output_path):
    
    os.makedirs(output_path, exist_ok=True)
    print("Running bilinear interpolation")

    mosaic_files = glob(os.path.join(input_path, "*.png"))

    for file in mosaic_files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not read {file}")
            continue

        # pretvorba BGR → RGB
        packed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pretvorba tri-kanalnog mozaika u jedan-kanalni
        bayer = np.zeros(packed.shape[:2], dtype=packed.dtype)
        mask = (packed[..., 0] > 0)
        bayer[mask] = packed[..., 0][mask]
        mask = (packed[..., 1] > 0)
        bayer[mask] = packed[..., 1][mask]
        mask = (packed[..., 2] > 0)
        bayer[mask] = packed[..., 2][mask]

        H, W = bayer.shape
        R = np.zeros_like(bayer)
        G = np.zeros_like(bayer)
        B = np.zeros_like(bayer)

        R_mask = np.zeros_like(bayer, dtype=bool)
        G_mask = np.zeros_like(bayer, dtype=bool)
        B_mask = np.zeros_like(bayer, dtype=bool)

        R_mask[0::2, 1::2] = True
        G_mask[0::2, 0::2] = True
        G_mask[1::2, 1::2] = True
        B_mask[1::2, 0::2] = True

        R[R_mask] = bayer[R_mask]
        G[G_mask] = bayer[G_mask]
        B[B_mask] = bayer[B_mask]

        def bilinear_fill(channel, mask, channel_type):

            # kernali za izracun preko konvolucije
            kernel_rb = 0.25 * np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]], dtype=float)
            kernel_g = 0.25 * np.array([[0, 1, 0],[1, 4, 1],[0, 1, 0]], dtype=float)

            if channel_type == 'G' : kernel = kernel_g 
            else : kernel = kernel_rb

            n = convolve(channel.astype(float), kernel, mode='mirror')

            out = n.astype(channel.dtype).copy()
            out[mask] = channel[mask]

            return out

        R_filled = bilinear_fill(R, R_mask, 'R')
        G_filled = bilinear_fill(G, G_mask, 'G')
        B_filled = bilinear_fill(B, B_mask, 'B')

        reconstructed = np.dstack([R_filled, G_filled, B_filled])
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # spremi u output
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(output_path, f"demosaic_{name}.png")

        cv2.imwrite(save_path, cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
        print(f"Saved reconstructed image: {save_path}")

    print("Done!")
