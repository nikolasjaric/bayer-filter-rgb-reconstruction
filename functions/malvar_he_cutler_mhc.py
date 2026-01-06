import os
import cv2
import numpy as np
from glob import glob
from scipy.ndimage import convolve


def malvar_he_cutler_mhc(input_path, output_path):
    
    os.makedirs(output_path, exist_ok=True)
    print("Running malvar_he_cutler_mhc")

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

        # koriste se 4 različita kernela (G na R/B, R/B na G horiz. i ver. te R na B i obratno)
        kernel_g_rb = (1/8) * np.array([
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  2,  0,  0],
            [-1,  2,  4,  2, -1],
            [ 0,  0,  2,  0,  0],
            [ 0,  0, -1,  0,  0]
        ])

        kernel_rb_gv = (1/8) * np.array([
            [ 0,  0, -1,  0,  0],
            [ 0, -1,  4, -1,  0],
            [0.5, 0,  5,  0,0.5],
            [ 0, -1,  4, -1,  0],
            [ 0,  0, -1,  0,  0]
        ])

        kernel_rb_gh = (1/8) * np.array([
            [ 0,  0,0.5,  0,  0],
            [ 0, -1,  0, -1,  0],
            [-1,  4,  5,  4, -1],
            [ 0, -1,  0, -1,  0],
            [ 0,  0,0.5,  0,  0]
        ])

        kernel_r_b = (1/8) * np.array([
            [ 0,  0,-11.5,  0,  0],
            [ 0,  2,  0,  2,  0],
            [-1.5,  0,  6,  0, -1.5],
            [ 0,  2,  0,  2,  0],
            [ 0,  0, -1.5,  0,  0]
        ])


        G_interp = convolve(G, kernel_g_rb, mode='mirror')
        G[~G_mask] = G_interp[~G_mask]

        R_h = convolve(R, kernel_rb_gh, mode='mirror')
        R_v = convolve(R, kernel_rb_gv, mode='mirror')
        R_b = convolve(R, kernel_r_b, mode='mirror')
        R[G_mask & (np.arange(H)[:,None] % 2 == 0)] = R_h[G_mask & (np.arange(H)[:,None] % 2 == 0)]
        R[G_mask & (np.arange(H)[:,None] % 2 == 1)] = R_v[G_mask & (np.arange(H)[:,None] % 2 == 1)]
        R[B_mask] = R_b[B_mask]

        B_h = convolve(B, kernel_rb_gh, mode='mirror')
        B_v = convolve(B, kernel_rb_gv, mode='mirror')
        B_r = convolve(B, kernel_r_b, mode='mirror')
        B[G_mask & (np.arange(H)[:,None] % 2 == 1)] = B_h[G_mask & (np.arange(H)[:,None] % 2 == 1)]
        B[G_mask & (np.arange(H)[:,None] % 2 == 0)] = B_v[G_mask & (np.arange(H)[:,None] % 2 == 0)]
        B[R_mask] = B_r[R_mask]

        reconstructed = np.dstack([R, G, B])
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # spremi u output
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(output_path, f"demosaic_{name}.png")

        cv2.imwrite(save_path, cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
        print(f"Saved reconstructed image: {save_path}")

    print("Done!")
