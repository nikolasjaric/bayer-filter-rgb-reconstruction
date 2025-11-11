import os
import cv2
import numpy as np
from glob import glob
from scipy.ndimage import distance_transform_edt


def nearest_neighbour(input_path, output_path):
    
    os.makedirs(output_path, exist_ok=True)
    print("Running nearest neighbour")

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

        # popunjavanje za R
        mask = (R == 0) # True gdje nedostaje vrijednost, False inace
        R_filled = R.copy()
        dist, indices = distance_transform_edt(mask, return_indices=True)
        R_filled[mask] = R[tuple(indices[:,mask])]

        # popunjavanje za G
        mask = (G == 0) 
        G_filled = G.copy()
        dist, indices = distance_transform_edt(mask, return_indices=True)
        G_filled[mask] = G[tuple(indices[:,mask])]

        # popunjavanje za B
        mask = (B == 0) 
        B_filled = B.copy()
        dist, indices = distance_transform_edt(mask, return_indices=True)
        B_filled[mask] = B[tuple(indices[:,mask])]

        reconstructed = np.dstack([R_filled, G_filled, B_filled])
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # spremi u output
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(output_path, f"demosaic_{name}.png")

        cv2.imwrite(save_path, cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
        print(f"Saved reconstructed image: {save_path}")

    print("Done!")
