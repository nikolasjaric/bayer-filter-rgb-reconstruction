import os
from glob import glob
import cv2
import numpy as np

def bayer_mosaic_generator(input_path, output_path):

    os.makedirs(output_path, exist_ok=True)
    print(f"Generating Bayer mosaic from: {input_path} to: {output_path}")

    search_pattern = os.path.join(input_path, '*.png')
    dataset = glob(search_pattern)

    for image in dataset:

        image_cv2 = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if image_cv2 is None:
            raise FileNotFoundError(f"Error: Could not load image: {image}")
        
        # File Naming
        base_name = os.path.basename(image) 
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(output_path, f"mosaic_{name}.png") # path and filename for saving

        color_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # extract height and width
        H, W, C = color_image.shape
        if C != 3:
            raise ValueError(f"Expected 3-channel RGB image, got {C} channels")


        # Initialize masks for the GRBG pattern
        G1_mask = np.zeros((H, W), dtype=bool) 
        R_mask = np.zeros((H, W), dtype=bool) 
        B_mask = np.zeros((H, W), dtype=bool) 
        G2_mask = np.zeros((H, W), dtype=bool) 

        # Create the masks (...[0::2] -> starting from 0 every other)
        G1_mask[0::2, 0::2] = True # G at even row, even col
        R_mask[0::2, 1::2] = True # R at even row, uneven col
        B_mask[1::2, 0::2] = True # B at uneven row, even col
        G2_mask[1::2, 1::2] = True # G at odd row, odd col

        # Combine the two green masks
        G_mask = G1_mask | G2_mask

        # Initialize image for storing mosaiced values
        packed = np.zeros((H, W, 3), dtype=color_image.dtype)

        # Extract values of pixel colors
        packed[..., 0][R_mask] = color_image[..., 0][R_mask]  # R
        packed[..., 1][G_mask] = color_image[..., 1][G_mask]  # G
        packed[..., 2][B_mask] = color_image[..., 2][B_mask]  # B

        # saving the 3-channel image
        ok = cv2.imwrite(save_path, cv2.cvtColor(packed, cv2.COLOR_RGB2BGR))
        if not ok:
            raise IOError(f"Failed to write: {save_path}")
