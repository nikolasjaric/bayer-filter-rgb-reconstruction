import numpy as np
import pandas as pd 
from glob import glob
import cv2
import matplotlib.pyplot as plt
import os


def bayer_mosaic_generator(input_path, output_path):
    print(f"Generating Bayer mosaic from: {input_path} to: {output_path}")

    search_pattern = os.path.join(input_path, '*.png')
    dataset = glob(search_pattern)

    for image in dataset:

        image_cv2 = cv2.imread(image)
        if image_cv2 is None:
            raise FileNotFoundError(f"Error: Could not load image")
        
        # File Naming
        base_name = os.path.basename(image) 
        name, ext = os.path.splitext(base_name)
        new_filename = f"mosaic_{name}.png" 
        save_path = os.path.join(output_path, new_filename) # path and filename for saving

        color_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # extract height and width
        H, W, C = color_image.shape

        # Initialize masks for the RGGB pattern
        R_mask = np.zeros((H, W), dtype=bool) # R at even row, uneven col
        G1_mask = np.zeros((H, W), dtype=bool) # G at even row, even col
        G2_mask = np.zeros((H, W), dtype=bool) # G at odd row, odd col
        B_mask = np.zeros((H, W), dtype=bool) # B at uneven row, even col

        # Create the masks (...[0::2] -> starting from 0 every other)
        R_mask[0::2, 1::2] = True
        G1_mask[0::2, 0::2] = True
        G2_mask[1::2, 1::2] = True
        B_mask[1::2, 0::2] = True

        # Combine G masks
        G_mask = G1_mask | G2_mask

        mosaiced_image = np.zeros((H, W), dtype=color_image.dtype)

        # Extract values of pixel colors
        mosaiced_image[R_mask] = color_image[..., 0][R_mask] 
        mosaiced_image[G_mask] = color_image[..., 1][G_mask]
        mosaiced_image[B_mask] = color_image[..., 2][B_mask]


        # DISPLAYING THE MOSAIC 
        # initialize a 3-channel (RGB) image for visualization so we can visualize all 3 colors again
        colored_bayer_pattern = np.zeros((H, W, 3), dtype=color_image.dtype)

        # Apply the captured Color values to the channel where mask is True
        colored_bayer_pattern[R_mask, 0] = color_image[R_mask, 0] # Red channel (index 0) -> on places where red is put intensity in the red channel
        colored_bayer_pattern[G_mask, 1] = color_image[G_mask, 1] # Green channel (index 1)
        colored_bayer_pattern[B_mask, 2] = color_image[B_mask, 2] # Blue channel (index 2)

        # saving the 3-channel image
        fig, ax = plt.subplots()
        ax.imshow(colored_bayer_pattern)
        ax.axis('off')
        save_path = save_path 
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    pass