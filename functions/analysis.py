from math import log10, sqrt
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import msssim
from skimage.color import deltaE_ciede2000 as ciede2000

def MSE(original, demosaiced):
    mse = np.mean((original - demosaiced) ** 2)
    return mse

def PSNR(original, demosaiced):
    mse = MSE(original, demosaiced)
    if(mse == 0):  # MSE is zero means no noise is present in the signal -> PSNR has no importance
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(original, demosaiced):
    ssim_score = ssim(original, demosaiced, win_size=None, channel_axis=-1)
    return ssim_score

# def SSIM(original, demosaiced):
#     mean_original = np.mean(original)
#     mean_demosaiced = np.mean(demosaiced)
#     std_original = np.std(original)
#     std_demosaiced = np.std(demosaiced)
#     covariance = np.mean((original - mean_original) * (demosaiced - mean_demosaiced))
#     c1 = (0.01 * 255) ** 2
#     c2 = (0.03 * 255) ** 2
#     c3 = c2 / 2

#     luminance = (2 * mean_original * mean_demosaiced + c1) / (mean_original ** 2 + mean_demosaiced ** 2 + c1)
#     contrast = (2 * std_original * std_demosaiced + c2) / (std_original ** 2 + std_demosaiced ** 2 + c2)
#     structure = (covariance + c3) / (std_original * std_demosaiced + c3)
#     ssim = luminance * contrast * structure
#     return ssim

def MS_SSIM(original, demosaiced):
    ms_ssim = msssim(original, demosaiced)
    return ms_ssim.real

def CIEDE2000(original, demosaiced):
    original = np.float32(original)
    original *= 1./255
    Lab1 = cv2.cvtColor(original, cv2.COLOR_BGR2Lab)

    demosaiced = np.float32(demosaiced)
    demosaiced *= 1./255
    Lab2 = cv2.cvtColor(demosaiced, cv2.COLOR_BGR2Lab)

    c2000 = ciede2000(Lab1, Lab2, channel_axis=-1)
    return c2000


def run_analysis (original_folder, reconstructed_folder, analysis_output_folder):
    originals = Path(original_folder).glob('*.png')
    reconstructeds = Path(reconstructed_folder).glob('*.png')
    file_name = analysis_output_folder+"/analysis.txt"
    f = open(file_name, "w")  

    for original, reconstructed in zip(originals,reconstructeds):
        image1 = cv2.imread(original)
        image2 = cv2.imread(reconstructed, 1)

        f.write(original.name+", "+reconstructed.name)
        value = MSE(image1, image2)
        f.write("\n MSE: " + str(value))
        value = PSNR(image1, image2)
        f.write("\n PSNR: " + str(value) + " dB")
        value = SSIM(image1, image2)
        f.write("\n SSIM: " + str(value))
        value = MS_SSIM(image1, image2)
        f.write("\n MS_SSIM: " + str(value))
        value = CIEDE2000(image1, image2)
        f.write("\n CIEDE2000: " + str(value[-1]) + "\n\n")

    f.close()
    return 