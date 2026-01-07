import os
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from glob import glob

def total_variation_regularization_tv(
    input_path,
    output_path,
    lambda_tv=0.1,
    alpha=0.1,
    epsilon=1e-4,
    max_iter=20
):

    os.makedirs(output_path, exist_ok=True)
    print(f"Running TV regularization demosaicing from {input_path} to {output_path}")

    image_files = [
        f for f in os.listdir(input_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for img_file in image_files:
        img_path = os.path.join(input_path, img_file)
        bayer = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bayer is None:
            print(f"Skipping {img_file}")
            continue

        # BGR → RGB
        bayer = cv2.cvtColor(bayer, cv2.COLOR_BGR2RGB).astype(np.float32)
        H, W, _ = bayer.shape

        # Bayer masks (GRBG)
        R_mask = np.zeros((H, W), dtype=bool)
        G_mask = np.zeros((H, W), dtype=bool)
        B_mask = np.zeros((H, W), dtype=bool)

        R_mask[0::2, 1::2] = True
        G_mask[0::2, 0::2] = True
        G_mask[1::2, 1::2] = True
        B_mask[1::2, 0::2] = True

        # Initialization using linear interpolation / inpainting
        I = np.zeros_like(bayer, dtype=np.float32)
        I[...,0][R_mask] = bayer[...,0][R_mask]
        I[...,1][G_mask] = bayer[...,1][G_mask]
        I[...,2][B_mask] = bayer[...,2][B_mask]

        for c in range(3):
            mask = np.zeros((H,W), dtype=np.uint8)
            if c == 0: mask[R_mask] = 1
            if c == 1: mask[G_mask] = 1
            if c == 2: mask[B_mask] = 1
            I[...,c] = cv2.inpaint(I[...,c].astype(np.float32), (1-mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)

        prev_energy = np.inf

        # TV + L2 optimization
        for k in range(max_iter):
            # TV proximal
            I_tv = denoise_tv_chambolle(I, weight=lambda_tv, max_num_iter=5)

            # L2 fidelity gradient
            grad = np.zeros_like(I)
            grad[...,0] = R_mask * (I[...,0] - bayer[...,0])
            grad[...,1] = G_mask * (I[...,1] - bayer[...,1])
            grad[...,2] = B_mask * (I[...,2] - bayer[...,2])

            # Gradient descent update
            I_new = I - alpha * (grad + (I - I_tv))

            # Energy for convergence check
            energy = (
                np.sum(R_mask * (I_new[...,0]-bayer[...,0])**2) +
                np.sum(G_mask * (I_new[...,1]-bayer[...,1])**2) +
                np.sum(B_mask * (I_new[...,2]-bayer[...,2])**2) +
                lambda_tv * np.sum(np.abs(I_new - I_tv))
            )

            if abs(prev_energy - energy) < epsilon:
                print(f"{img_file}: converged after {k+1} iterations")
                break

            prev_energy = energy
            I = I_new

        # Save result
        I_out = np.clip(I, 0, 255).astype(np.uint8)
        save_path = os.path.join(output_path, img_file)
        cv2.imwrite(save_path, cv2.cvtColor(I_out, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}")

    print("TV demosaicing done.")
