from math import log10, sqrt
import cv2
import numpy as np
from pathlib import Path

def MSE(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    return mse

def PSNR(original, compressed):
    mse = MSE(original, compressed)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# get the path/directory
def run_analysis (original_folder, reconstructed_folder, analysis_output_folder, method):
    originals = Path(original_folder).glob('*.png')
    reconstructeds = Path(reconstructed_folder).glob('*.png')
    values_mse = []
    values_psnr = []
    # ... jos za druge analize
    file_name = analysis_output_folder+"/analysis.txt"
    f = open(file_name, "w")

    if method == "Nearest Neighbour":
        method = "nearest_neighbour"
    elif method == "Bilinear Interpolation":
        method = "bilinear_interpolation"
    elif method == "Bicubic Interpolation":
        method = "bicubic_interpolation"
    elif method == "Malvar-He-Cutler (MHC)":
        method = "malvar_he_cutler_mhc"
    elif method == "Frequency Reconstruction (Fourier)":
        method = "frequency_reconstruction_fourier"
    elif method == "Total Variation Regularization (TV)":
        method = "total_variation_regularization_tv"
    elif method == "CNN-Based Reconstruction":
        method = "cnn_based_reconstruction"
    

    for original in originals:
        image = cv2.imread(original)
        for reconstructed in reconstructeds:
            if (reconstructed.name == str(method)+"_"+original.name):
                compressed = cv2.imread(reconstructed, 1)
                f.write(original.name)

                value = MSE(image, compressed)
                values_mse.append(value)
                f.write("\n MSE: " + str(value))
                value = PSNR(image, compressed)
                values_psnr.append(value)
                f.write("\n PSNR: " + str(value) + " dB")
        # ... jos za druge analize
    f.close()


# if __name__ == "__main__":
#     main()