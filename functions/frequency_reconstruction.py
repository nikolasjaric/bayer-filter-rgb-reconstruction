"""

This file implements frequency-domain Bayer demosaicing as part of a larger project
focused on reconstructing full-color images from Bayer Color Filter Array (CFA) data.

METHOD OVERVIEW
This file implements a frequency-domain approach to Bayer demosaicing. The core idea is:
1. Separate the mosaiced image into luminance (brightness) and chrominance (color) components
   using Fourier analysis.
2. Interpolate the color information using low-frequency characteristics.
3. Recombine luminance and color to reconstruct full RGB images.

This approach leverages the fact that luminance and chrominance occupy different regions
of the frequency spectrum, making them separable via frequency-domain filtering.

"""

import os
from glob import glob
import cv2
import numpy as np

def _rggb_masks(shape):
    """
    Creating binary masks identifying pixel positions for each color in the GRBG Bayer pattern.
    GRBG pattern:
      G R
      B G
    
    Returns three boolean masks (R_mask, G_mask, B_mask) that identify where each
    color channel is sampled in the CFA.
    """
    H, W = shape
    R_mask = np.zeros((H, W), dtype=bool)
    G_mask = np.zeros((H, W), dtype=bool)
    B_mask = np.zeros((H, W), dtype=bool)

    # GRBG pattern
    G_mask[0::2, 0::2] = True  # even row, even col
    R_mask[0::2, 1::2] = True  # even row, odd col
    B_mask[1::2, 0::2] = True  # odd row, even col
    G_mask[1::2, 1::2] = True  # odd row, odd col

    return R_mask, G_mask, B_mask

def _bayer_from_packed_rgb(packed_rgb: np.ndarray) -> np.ndarray:
    """
    Convert a packed RGB representation of the Bayer mosaic --> single-channel CFA image.
    
    A "packed" image has R, G, and B values in their respective channel positions,
    with zeros at positions where that color is not sampled (due to the Bayer pattern).
    This function combines all three channels into a single grayscale image where each
    pixel contains whichever color was sampled at that location.
    
    Args:
        packed_rgb: 3-channel array in RGB order, representing the Bayer mosaic.
    
    Returns:
        Single-channel CFA image (H x W).
    """
    if packed_rgb.ndim != 3 or packed_rgb.shape[2] < 3:
        raise ValueError("packed_rgb must be HxWx3 (or HxWx4) array")

    # Extract first 3 channels (R, G, B in RGB order)
    rgb = packed_rgb[..., :3]
    H, W, _ = rgb.shape
    bayer = np.zeros((H, W), dtype=rgb.dtype)

    R_mask, G_mask, B_mask = _rggb_masks((H, W))

    bayer[R_mask] = rgb[..., 0][R_mask]  # Channel 0 = R
    bayer[G_mask] = rgb[..., 1][G_mask]  # Channel 1 = G
    bayer[B_mask] = rgb[..., 2][B_mask]  # Channel 2 = B

    return bayer

def _estimate_luminance_fourier(cfa: np.ndarray, cutoff: float = None) -> np.ndarray:
	"""
	Extract luminance (brightness) using a Fourier low-pass on the sampled CFA,
	then build per-color low-frequency maps and combine them using sampling-aware
	weights. This follows the practical advice in the references: compensate for
	the GRBG sampling density when combining per-channel low-frequency estimates.
	"""
	if cutoff is None:
		cutoff = 0.45

	cfa_f = cfa.astype(np.float64)
	H, W = cfa_f.shape

	# Pad with reflection to reduce FFT edge artifacts
	cfa_padded = cv2.copyMakeBorder(cfa_f, H//4, H//4, W//4, W//4, cv2.BORDER_REFLECT)
	H_pad, W_pad = cfa_padded.shape

	F = np.fft.fft2(cfa_padded)
	F = np.fft.fftshift(F)

	u = np.fft.fftfreq(W_pad)
	v = np.fft.fftfreq(H_pad)
	u = np.fft.fftshift(u)
	v = np.fft.fftshift(v)
	U, V = np.meshgrid(u, v)
	R = np.sqrt(U**2 + V**2)

	sigma = cutoff / np.sqrt(2 * np.log(2))
	H_low = np.exp(-(R**2) / (2 * sigma**2))

	F_lum = F * H_low
	F_lum = np.fft.ifftshift(F_lum)
	lum_padded = np.fft.ifft2(F_lum).real

	lum = lum_padded[H//4:H//4+H, W//4:W//4+W]

	R_mask, G_mask, B_mask = _rggb_masks((H, W))

	lum_R = np.zeros_like(lum)
	lum_G = np.zeros_like(lum)
	lum_B = np.zeros_like(lum)

	lum_R[R_mask] = lum[R_mask]
	lum_G[G_mask] = lum[G_mask]
	lum_B[B_mask] = lum[B_mask]

	lum_R_full = cv2.GaussianBlur(lum_R, (3, 3), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_REFLECT)
	lum_G_full = cv2.GaussianBlur(lum_G, (3, 3), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_REFLECT)
	lum_B_full = cv2.GaussianBlur(lum_B, (3, 3), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_REFLECT)

	# Sampling-density-adjusted perceptual weights (ITU-R reference scaled by sampling density).
	# This preserves perceived luminance while compensating for G oversampling.
	# p = original perceptual weight; density = sampling density per Bayer pattern position.
	pR = 0.2125; pG = 0.7154; pB = 0.0721
	dR = 0.25; dG = 0.50; dB = 0.25

	pR_adj = pR / dR
	pG_adj = pG / dG
	pB_adj = pB / dB

	# Normalize to unity gain so luminance mean is preserved (avoid global brightness shift).
	sum_adj = (pR_adj + pG_adj + pB_adj)
	pR_norm = pR_adj / sum_adj
	pG_norm = pG_adj / sum_adj
	pB_norm = pB_adj / sum_adj

	# Combine low-frequency (luminance) maps with density-corrected perceptual weights.
	# This follows Dubois guidance: combine per-channel low-frequency content with proper normalization.
	lum_weighted = pR_norm * lum_R_full + pG_norm * lum_G_full + pB_norm * lum_B_full

	return lum_weighted.astype(np.float64)

def _interpolate_chrominance(ch: np.ndarray, ksize: int = 7, sigma: float = 1.5) -> np.ndarray:
	"""
	Fill missing color information across the image using Gaussian interpolation.
	
	Since each pixel in a Bayer mosaic samples only one color, the other two colors
	are missing at that location. This function interpolates missing values by
	applying Gaussian blur, which treats nearby sampled values as evidence for
	the missing color in the neighborhood.
	
	Args:
		ch: Single-color channel with values at sampled positions (others are zero).
		ksize: Kernel size for Gaussian blur (larger = more smoothing).
		sigma: Standard deviation for Gaussian blur (higher = smoother).
	
	Returns:
		Full-resolution color channel with interpolated values everywhere.
	"""
	ch_f = ch.astype(np.float64)
	if ksize % 2 == 0:
		ksize += 1
	ch_interp = cv2.GaussianBlur(
		ch_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
	)
	return ch_interp

def _demosaic_fourier(cfa: np.ndarray) -> np.ndarray:
	"""
	 demosaicing:
	- Compute chrominance residuals as before.
	- After interpolating chroma, scale each chroma channel so its full-resolution
	  std matches the std of the sampled residuals. This preserves sampled chroma energy
	  (prevents chroma attenuation caused by interpolation smoothing).
	"""
	H, W = cfa.shape

	# Step 1: Extract luminance
	Y = _estimate_luminance_fourier(cfa, cutoff=0.45)

	# Step 2: Chrominance residual
	C_scalar = cfa.astype(np.float64) - Y

	# Step 3: Demultiplex by Bayer positions
	R_mask, G_mask, B_mask = _rggb_masks((H, W))

	C_R = np.zeros((H, W), dtype=np.float64)
	C_G = np.zeros((H, W), dtype=np.float64)
	C_B = np.zeros((H, W), dtype=np.float64)

	C_R[R_mask] = C_scalar[R_mask]
	C_G[G_mask] = C_scalar[G_mask]
	C_B[B_mask] = C_scalar[B_mask]

	# Step 4: Interpolate chrominance
	C_R_full = _interpolate_chrominance(C_R, ksize=7, sigma=1.5)
	C_G_full = _interpolate_chrominance(C_G, ksize=7, sigma=1.5)
	C_B_full = _interpolate_chrominance(C_B, ksize=7, sigma=1.5)

	# ---- Chroma energy restoration (minimal, principled) ----
	# Compute std at sampled positions (what sensor actually measured) and full interpolated std.
	eps = 1e-8
	std_sample_R = C_R[R_mask].std() if np.any(R_mask) else 0.0
	std_sample_G = C_G[G_mask].std() if np.any(G_mask) else 0.0
	std_sample_B = C_B[B_mask].std() if np.any(B_mask) else 0.0

	std_full_R = C_R_full.std()
	std_full_G = C_G_full.std()
	std_full_B = C_B_full.std()

	# Scale each interpolated chroma channel so its std matches the sampled std (prevents attenuation)
	if std_full_R > eps and std_sample_R > 0:
		C_R_full *= (std_sample_R / std_full_R)
	if std_full_G > eps and std_sample_G > 0:
		C_G_full *= (std_sample_G / std_full_G)
	if std_full_B > eps and std_sample_B > 0:
		C_B_full *= (std_sample_B / std_full_B)

	# Step 5: Reconstruct RGB
	R = Y + C_R_full
	G = Y + C_G_full
	B = Y + C_B_full

	# Final clamp, return uint8
	channels = np.stack([R, G, B], axis=-1)
	channels_clipped = np.clip(channels, 0, 255)

	rgb = channels_clipped.astype(np.uint8)
	return rgb


def frequency_reconstruction(input_path: str, output_path: str) -> None:
    
    os.makedirs(output_path, exist_ok=True)
    mosaic_files = glob(os.path.join(input_path, "*.png"))
    
    for file in mosaic_files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Support 3-channel and 4-channel input; if grayscale convert to RGB
        if img.ndim == 2:
            packed_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            packed_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            packed_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            continue

        try:
            cfa = _bayer_from_packed_rgb(packed_rgb)
        except ValueError:
            continue

        reconstructed_rgb = _demosaic_fourier(cfa)
        
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(output_path, f"frequency_demosaic_{name}.png")

        cv2.imwrite(save_path, cv2.cvtColor(reconstructed_rgb, cv2.COLOR_RGB2BGR))

print("\nFrequency Reconstruction DONE.")