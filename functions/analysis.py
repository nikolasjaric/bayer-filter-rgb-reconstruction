from math import log10, sqrt
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import msssim
from skimage.color import deltaE_ciede2000 as ciede2000

def MSE(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    return mse

def PSNR(original, compressed):
    mse = MSE(original, compressed)
    if(mse == 0):  # MSE is zero means no noise is present in the signal -> PSNR has no importance
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(original, compressed):
    ssim_score = ssim(original, compressed, win_size=None, channel_axis=-1)
    return ssim_score

# def SSIM(original, compressed):
#     mean_original = np.mean(original)
#     mean_compressed = np.mean(compressed)
#     std_original = np.std(original)
#     std_compressed = np.std(compressed)
#     covariance = np.mean((original - mean_original) * (compressed - mean_compressed))
#     c1 = (0.01 * 255) ** 2
#     c2 = (0.03 * 255) ** 2
#     c3 = c2 / 2

#     luminance = (2 * mean_original * mean_compressed + c1) / (mean_original ** 2 + mean_compressed ** 2 + c1)
#     contrast = (2 * std_original * std_compressed + c2) / (std_original ** 2 + std_compressed ** 2 + c2)
#     structure = (covariance + c3) / (std_original * std_compressed + c3)
#     ssim = luminance * contrast * structure
#     return ssim

def MS_SSIM(original, compressed):
    ms_ssim = msssim(original, compressed)
    return ms_ssim.real

def CIEDE2000(original, compressed):
    original = np.float32(original)
    original *= 1./255
    Lab1 = cv2.cvtColor(original, cv2.COLOR_BGR2Lab)

    compressed = np.float32(compressed)
    compressed *= 1./255
    Lab2 = cv2.cvtColor(compressed, cv2.COLOR_BGR2Lab)

    c2000 = ciede2000(Lab1, Lab2, channel_axis=-1)
    return c2000


def run_analysis (original_folder, reconstructed_folder, analysis_output_folder, method):
    originals = Path(original_folder).glob('*.png')
    reconstructeds = Path(reconstructed_folder).glob('*.png')
    file_name = analysis_output_folder+"/analysis.txt"
    f = open(file_name, "w")  

    for original, reconstructed in zip(originals,reconstructeds):
        image = cv2.imread(original)
        compressed = cv2.imread(reconstructed, 1)

        f.write(original.name+", "+reconstructed.name)
        value = MSE(image, compressed)
        f.write("\n MSE: " + str(value))
        value = PSNR(image, compressed)
        f.write("\n PSNR: " + str(value) + " dB")
        value = SSIM(image, compressed)
        f.write("\n SSIM: " + str(value))
        value = MS_SSIM(image, compressed)
        f.write("\n MS_SSIM: " + str(value))
        value = CIEDE2000(image, compressed)
        f.write("\n CIEDE2000: " + str(value) + "\n\n")

    f.close()
    return 