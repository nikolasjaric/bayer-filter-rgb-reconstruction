import os
import cv2
import numpy as np
from glob import glob
from scipy.ndimage import convolve


def bicubic_interpolation(input_path, output_path):
    
    os.makedirs(output_path, exist_ok=True)
    print("Running bicubic interpolation")

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

        Fg = (1/256) * np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, -9, 0, -9, 0, 0],
            [0, -9, 0, 81, 0, -9, 0],
            [1, 0, 81, 256, 81, 0, 1],
            [0, -9, 0, 81, 0, -9, 0],
            [0, 0, -9, 0, -9, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        Fc = (1/256) * np.array([
            [1, 0, -9, -16, -9, 0, 1],
            [0, 0, 0,   0,  0, 0, 0],
            [-9, 0, 81, 144, 81, 0, -9],
            [-16, 0, 144, 256, 144, 0, -16],
            [-9, 0, 81, 144, 81, 0, -9],
            [0, 0, 0,   0,  0, 0, 0],
            [1, 0, -9, -16, -9, 0, 1]
        ], dtype=np.float32)

        # Fg za zelenu, Fc za crvenu i plavu
        G_interp = cv2.filter2D(G, -1, Fg)
        R_interp = cv2.filter2D(R, -1, Fc)
        B_interp = cv2.filter2D(B, -1, Fc)

        G_filled = G.copy()
        R_filled = R.copy()
        B_filled = B.copy()
        G_filled[~G_mask] = G_interp[~G_mask]
        R_filled[~R_mask] = R_interp[~R_mask]
        B_filled[~B_mask] = B_interp[~B_mask]

        # losija verzija, ali ne koristi gotovu funkciju (cv2.fillter2D)
        """def bicubic_fill(channel, mask):

            w = np.array([-1/16, 9/16, 9/16, -1/16])
            kernel = np.outer(w,w)

            n = convolve(channel.astype(float), kernel, mode='mirror')
            m = convolve(mask.astype(float), kernel, mode='mirror')

            interpolated = n / m

            out = interpolated.astype(channel.dtype).copy()
            out[mask] = channel[mask]

            return out

        
        R_filled = bicubic_fill(R, R_mask)
        G_filled = bicubic_fill(G, G_mask)
        B_filled = bicubic_fill(B, B_mask)"""

        reconstructed = np.dstack([R_filled, G_filled, B_filled])
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # spremi u output
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(output_path, f"demosaic_{name}.png")

        cv2.imwrite(save_path, cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
        print(f"Saved reconstructed image: {save_path}")

    print("Done!")
