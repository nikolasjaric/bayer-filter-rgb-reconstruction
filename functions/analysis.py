from math import log10, sqrt
import cv2
import numpy as np
from pathlib import Path

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


def run_analysis (original_folder, reconstructed_folder, analysis_output_folder, method):
    originals = Path(original_folder).glob('*.png')
    reconstructeds = Path(reconstructed_folder).glob('*.png')
    file_name = analysis_output_folder+"/analysis.txt"
    f = open(file_name, "w")

    method = method.lower()
    method = method.replace("-", " ")
    method = method.replace("(", "")
    method = method.replace(")", "")
    m = method.split(" ")
    method = "_".join(m)    

    for original in originals:
        image = cv2.imread(original)
        for reconstructed in reconstructeds:
            if (reconstructed.name == str(method)+"_"+original.name):
                compressed = cv2.imread(reconstructed, 1)

                f.write(original.name)
                value = MSE(image, compressed)
                f.write("\n MSE: " + str(value))
                value = PSNR(image, compressed)
                f.write("\n PSNR: " + str(value) + " dB")
                # ...
    f.close()
    return 