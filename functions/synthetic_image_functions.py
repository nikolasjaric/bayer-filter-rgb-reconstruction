import numpy as np
import matplotlib.image as mpimg
import os

def generate_synthetic_images(output_folder, img_size=(256, 256)):
    """
    Generates and saves three synthetic RGB images:
      1. A red-green diagonal checkerboard pattern.
      2. Geometric shapes (square + circle).
      3. Same as (2) with added Gaussian noise.
    
    Args:
        output_folder (str): Folder to save the generated images.
        img_size (tuple): Image size (height, width). Default = (256, 256).
    
    Returns:
        list: [checkerboard_img, shapes_img, noisy_img]
    """

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    height, width = img_size

    # 1️⃣ Checkerboard pattern
    def checkerboard(height, width, line_thickness=4):
        color1 = np.array([255, 0, 0], dtype=np.uint8)  # red
        color2 = np.array([0, 255, 0], dtype=np.uint8)  # green
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if ((x + y) // line_thickness) % 2 == 0:
                    img[y, x] = color1
                else:
                    img[y, x] = color2
        return img

    # 2️⃣ Geometric shapes
    def geometric_shapes(height, width):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # red square
        img[50:150, 50:150] = [255, 0, 0]
        # green circle
        Y, X = np.ogrid[:height, :width]
        mask = (X - 180) ** 2 + (Y - 180) ** 2 <= 30 ** 2
        img[mask] = [0, 255, 0]
        return img

    # 3️⃣ Add Gaussian noise
    def add_noise(img, snr=20):
        noise = np.random.normal(0, 255 / snr, img.shape)
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noisy_img

    # Generate images
    img1 = checkerboard(height, width)
    img2 = geometric_shapes(height, width)
    img3 = add_noise(img2, snr=5)

    # Save images
    mpimg.imsave(os.path.join(output_folder, "synthetic_checkerboard.png"), img1)
    mpimg.imsave(os.path.join(output_folder, "synthetic_shapes.png"), img2)
    mpimg.imsave(os.path.join(output_folder, "synthetic_shapes_noise.png"), img3)

    print(f"Synthetic images saved in folder: {output_folder} ✅")

    return [img1, img2, img3]
