#frequency reconstruction with Alleysson–Süsstrunk–Hérault method
import os
from glob import glob
import cv2
import numpy as np

def _rggb_masks(shape):
    """
    Generates logical masks for the GRBG Bayer pattern.
    This matches the pattern used in bayer_mosaic_generator():
    G R
    B G
    Despite the function name (_rggb_masks), this project uses GRBG pattern.
    """
    H, W = shape
    R_mask = np.zeros((H, W), dtype=bool)
    G_mask = np.zeros((H, W), dtype=bool)
    B_mask = np.zeros((H, W), dtype=bool)

    # GRBG pattern (matching bayer_mosaic_generator in mosaic_functions.py)
    G_mask[0::2, 0::2] = True  # even row, even col
    R_mask[0::2, 1::2] = True  # even row, odd col
    B_mask[1::2, 0::2] = True  # odd row, even col
    G_mask[1::2, 1::2] = True  # odd row, odd col

    return R_mask, G_mask, B_mask

def _bayer_from_packed_rgb(packed_rgb: np.ndarray) -> np.ndarray:
    """
    Convert a 'packed' 3-channel Bayer mosaic (R/G/B present only at their positions, others 0)
    into a single-channel Bayer CFA signal (scalar per pixel).
    
    *packed_rgb is expected to be in RGB channel order (not BGR).
    """
    if packed_rgb.ndim != 3 or packed_rgb.shape[2] < 3:
        raise ValueError("packed_rgb must be HxWx3 (or HxWx4) array")

    # Extract first 3 channels (R, G, B in RGB order)
    rgb = packed_rgb[..., :3]
    H, W, _ = rgb.shape
    bayer = np.zeros((H, W), dtype=rgb.dtype)

    R_mask, G_mask, B_mask = _rggb_masks((H, W))

    # GRBG Bayer pattern: R at (even_row, odd_col), G at (even_row, even_col) & (odd_row, odd_col), B at (odd_row, even_col)
    # Assume packed_rgb has R in channel 0, G in channel 1, B in channel 2 (RGB convention)
    bayer[R_mask] = rgb[..., 0][R_mask]  # Channel 0 = R
    bayer[G_mask] = rgb[..., 1][G_mask]  # Channel 1 = G
    bayer[B_mask] = rgb[..., 2][B_mask]  # Channel 2 = B

    return bayer

def _estimate_luminance_fourier(cfa: np.ndarray, cutoff: float = 0.45) -> np.ndarray:
    """ Estimate luminance from a CFA image via a low-pass filter in the Fourier domain.
    Uses perceptual RGB weighting to account for human color sensitivity and GRBG sampling parity.
    """
    cfa_f = cfa.astype(np.float64)
    H, W = cfa_f.shape
    
    # Apply periodic boundary conditions (FFT assumption) to avoid edge artifacts
    # Pad with reflection before FFT
    cfa_padded = cv2.copyMakeBorder(cfa_f, H//4, H//4, W//4, W//4, cv2.BORDER_REFLECT)
    H_pad, W_pad = cfa_padded.shape
    
    F = np.fft.fft2(cfa_padded)
    F = np.fft.fftshift(F)  # Shift DC component to the center
    
    # Normalized frequencies
    u = np.fft.fftfreq(W_pad)
    v = np.fft.fftfreq(H_pad)
    u = np.fft.fftshift(u)
    v = np.fft.fftshift(v)
    U, V = np.meshgrid(u, v)
    R = np.sqrt(U**2 + V**2)
    
    # Gaussian low-pass filter: sigma adjusted to the cutoff (FWHM relation)
    sigma = cutoff / np.sqrt(2 * np.log(2))
    H_low = np.exp(-(R**2) / (2 * sigma**2))
    
    F_lum = F * H_low
    F_lum = np.fft.ifftshift(F_lum)  # Shift back before inverse transform
    lum_padded = np.fft.ifft2(F_lum).real
    
    # Remove padding
    lum = lum_padded[H//4:H//4+H, W//4:W//4+W]
    
    
    R_mask, G_mask, B_mask = _rggb_masks((H, W))
    
    lum_R = np.zeros_like(lum)
    lum_G = np.zeros_like(lum)
    lum_B = np.zeros_like(lum)
    
    lum_R[R_mask] = lum[R_mask]
    lum_G[G_mask] = lum[G_mask]
    lum_B[B_mask] = lum[B_mask]
    
    # Interpolate each channel (light blur to preserve low-freq structure)
    lum_R_full = cv2.GaussianBlur(lum_R, (3, 3), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_REFLECT)
    lum_G_full = cv2.GaussianBlur(lum_G, (3, 3), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_REFLECT)
    lum_B_full = cv2.GaussianBlur(lum_B, (3, 3), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_REFLECT)
    
    # Combine with sampling-density-adjusted weights (ITU-R BT.709 perceptual weights,
    # scaled inversely by GRBG sampling density to preserve energy balance).
    # Standard weights: pR=0.2125, pG=0.7154, pB=0.0721
    # Sampling density: R=0.25, G=0.50, B=0.25
    # Adjusted: pR_adj=0.85, pG_adj=1.43, pB_adj=0.29 (relative)
    # Normalized sum to preserve mean luminance: divide by (0.85+1.43+0.29) ≈ 2.57
    pR_adj = 0.2125 / 0.25 / 2.57  # ≈ 0.330
    pG_adj = 0.7154 / 0.50 / 2.57  # ≈ 0.556
    pB_adj = 0.0721 / 0.25 / 2.57  # ≈ 0.112
    
    lum_weighted = pR_adj * lum_R_full + pG_adj * lum_G_full + pB_adj * lum_B_full
    
    return lum_weighted.astype(np.float64)

def _interpolate_chrominance(ch: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Chrominance interpolation: Gaussian blur with BORDER_REFLECT for consistency
    with the Alleysson method (avoiding edge discontinuities).
    
    ch: subsampled chrominance channel (only at R/G/B positions, others 0).
    """
    ch_f = ch.astype(np.float64)
    if ksize % 2 == 0:
        ksize += 1
    # Use BORDER_REFLECT instead of default border to preserve interpolation integrity
    ch_interp = cv2.GaussianBlur(
        ch_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )
    return ch_interp

def _check_array_health(arr: np.ndarray, name: str = "array") -> None:
    """Diagnostic: Check for NaN, Inf, dtype, and value range; print std as well."""
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    min_val = np.nanmin(arr) if arr.size > 0 else None
    max_val = np.nanmax(arr) if arr.size > 0 else None
    mean_val = np.nanmean(arr) if arr.size > 0 else None
    std_val = np.nanstd(arr) if arr.size > 0 else None
    
    print(f"  [{name}] dtype={arr.dtype}, shape={arr.shape}, NaN={nan_count}, Inf={inf_count}, "
          f"min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")
    
    if nan_count > 0 or inf_count > 0:
        print(f"    WARNING: {name} contains NaN/Inf values; clamping...")
        arr[~np.isfinite(arr)] = 0.0

def _demosaic_alleysson_fourier(cfa: np.ndarray) -> np.ndarray:
    """
    Alleysson–Süsstrunk–Hérault demosaicing kernel:
    1) Luminance Y = low-pass filtered CFA (frequency separation).
    2) Chrominance C = CFA - Y (high-frequency component).
    3) Demultiplex C by GRBG masks into three chrominance channels.
    4) Apply color-difference gains to account for GRBG sampling bias:
       - Green is sampled 2x per 2×2 block (no gain needed).
       - Red and Blue each sampled 1x (apply gain 2.0 to recover full-resolution chroma).
    5) Interpolate each chrominance channel (Gaussian blur).
    6) Reconstruction: R=Y+C_R, G=Y+C_G, B=Y+C_B (additive model).
    
    Note: GRBG Bayer pattern samples green at 50%, red at 25%, blue at 25%.
    To restore color fidelity, we apply inverse-sampling-frequency scaling before interpolation.
    """
    H, W = cfa.shape
    
    # 1) Luminance: Fourier-based frequency separation; a cutoff of 0.45 maintains finer
    # color information while still separating luminance from fine-scale chrominance.
    print("  [Luminance] Computing Fourier-based low-pass filter...")
    Y = _estimate_luminance_fourier(cfa, cutoff=0.45)
    _check_array_health(Y, "Luminance Y")
    
    # 2) Chrominance as residual
    print("  [Chrominance] Extracting residual C = CFA - Y...")
    C_scalar = cfa.astype(np.float64) - Y
    _check_array_health(C_scalar, "Residual C")
    
    # 3) Demultiplex chrominance by GRBG positions
    print("  [Demultiplex] Separating R, G, B chrominance channels...")
    R_mask, G_mask, B_mask = _rggb_masks((H, W))
    
    C_R = np.zeros((H, W), dtype=np.float64)
    C_G = np.zeros((H, W), dtype=np.float64)
    C_B = np.zeros((H, W), dtype=np.float64)
    
    C_R[R_mask] = C_scalar[R_mask]
    C_G[G_mask] = C_scalar[G_mask]
    C_B[B_mask] = C_scalar[B_mask]
    
    _check_array_health(C_R, "C_R (sampled, pre-gain)")
    _check_array_health(C_G, "C_G (sampled, pre-gain)")
    _check_array_health(C_B, "C_B (sampled, pre-gain)")
    
    # Quick diagnostics: pre-gain chroma-to-luma energy ratios
    C_R_sample = C_R[R_mask]
    C_G_sample = C_G[G_mask]
    C_B_sample = C_B[B_mask]
    Y_std = np.nanstd(Y) if Y.size else 0.0
    C_R_std_pre = np.nanstd(C_R_sample) if C_R_sample.size else 0.0
    C_G_std_pre = np.nanstd(C_G_sample) if C_G_sample.size else 0.0
    C_B_std_pre = np.nanstd(C_B_sample) if C_B_sample.size else 0.0
    if Y_std > 1e-12:
        print(f"  [Diagnostics pre-gain] std(Y)={Y_std:.4f}, std(C_R)={C_R_std_pre:.4f}, std(C_G)={C_G_std_pre:.4f}, std(C_B)={C_B_std_pre:.4f}")
        print(f"  [Diagnostics pre-gain] Chroma/Luma ratios: R={C_R_std_pre/Y_std:.4f}, G={C_G_std_pre/Y_std:.4f}, B={C_B_std_pre/Y_std:.4f}")
    
    # 4) Apply color-difference gains to account for GRBG sampling bias.
    # GRBG Bayer pattern:
    #   G(50%) R(25%)
    #   B(25%) G(50%)
    # REMOVED: Incorrect sparse-domain gain amplification.
    # Proper handling is via perceptual luminance weighting (now in _estimate_luminance_fourier).
    # The residual chrominance C = CFA - Y is already correctly balanced by the luminance model.
    print("  [Note] Chrominance balance handled via perceptual luminance weighting.")
    
    _check_array_health(C_R, "C_R (post-gain)")
    _check_array_health(C_G, "C_G (post-gain)")
    _check_array_health(C_B, "C_B (post-gain)")
    
    # Post-gain diagnostics
    C_R_std_post = np.nanstd(C_R[R_mask]) if C_R[R_mask].size else 0.0
    C_B_std_post = np.nanstd(C_B[B_mask]) if C_B[B_mask].size else 0.0
    if Y_std > 1e-12:
        print(f"  [Diagnostics post-gain] std(C_R)={C_R_std_post:.4f}, std(C_B)={C_B_std_post:.4f}")
        print(f"  [Diagnostics post-gain] Chroma/Luma ratios: R={C_R_std_post/Y_std:.4f}, B={C_B_std_post/Y_std:.4f}")
    
    # 5) Chrominance interpolation (Gaussian blur with reflective border)
    print("  [Interpolate] Gaussian blur on chrominance channels...")
    C_R_full = _interpolate_chrominance(C_R, ksize=5, sigma=1.0)
    C_G_full = _interpolate_chrominance(C_G, ksize=5, sigma=1.0)
    C_B_full = _interpolate_chrominance(C_B, ksize=5, sigma=1.0)
    
    _check_array_health(C_R_full, "C_R_full (interpolated)")
    _check_array_health(C_G_full, "C_G_full (interpolated)")
    _check_array_health(C_B_full, "C_B_full (interpolated)")
    
    # Post-interpolation diagnostics
    C_R_std_interp = np.nanstd(C_R_full)
    C_G_std_interp = np.nanstd(C_G_full)
    C_B_std_interp = np.nanstd(C_B_full)
    if Y_std > 1e-12:
        print(f"  [Diagnostics post-interp] std(C_R)={C_R_std_interp:.4f}, std(C_G)={C_G_std_interp:.4f}, std(C_B)={C_B_std_interp:.4f}")
        print(f"  [Diagnostics post-interp] Chroma/Luma ratios: R={C_R_std_interp/Y_std:.4f}, G={C_G_std_interp/Y_std:.4f}, B={C_B_std_interp/Y_std:.4f}")
    
    # 6) Reconstruct RGB: additive combination of luminance and interpolated chrominance
    print("  [Reconstruct] RGB = Y + C channels...")
    R = Y + C_R_full
    G = Y + C_G_full
    B = Y + C_B_full
    
    _check_array_health(R, "R_raw (pre-clip)")
    _check_array_health(G, "G_raw (pre-clip)")
    _check_array_health(B, "B_raw (pre-clip)")
    
    # Print per-channel balance:
    R_mean = np.nanmean(R)
    G_mean = np.nanmean(G)
    B_mean = np.nanmean(B)
    print(f"  [Output Balance] mean(R)={R_mean:.4f}, mean(G)={G_mean:.4f}, mean(B)={B_mean:.4f}")
    print(f"  [Output Balance] R/G ratio: {R_mean / (G_mean + 1e-12):.4f} (target: 0.85–1.0)")
    print(f"  [Output Balance] B/G ratio: {B_mean / (G_mean + 1e-12):.4f} (target: 0.75–0.95)")
    
    # Normalize/clamp to valid range
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)
    
    rgb = np.stack([R, G, B], axis=-1)
    rgb = rgb.astype(np.uint8)
    
    _check_array_health(rgb.astype(np.float64), "RGB_final")
    
    return rgb


def frequency_reconstruction_alleysson(input_path: str, output_path: str) -> None:
    """
    Main function for frequency-based Bayer demosaicing using Alleysson method.
    input_path: folder containing .png Bayer mosaics (3-channel 'packed').
    output_path: folder for reconstructed RGB images (demosaic_<name>.png).
    """
    os.makedirs(output_path, exist_ok=True)
    mosaic_files = glob(os.path.join(input_path, "*.png"))
    print(f"Processing from: {input_path} to: {output_path}")
    
    for file in mosaic_files:
        print(f"\nProcessing: {file}")
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not read {file}")
            continue

        # Support 3-channel and 4-channel input; if grayscale convert to RGB
        if img.ndim == 2:
            packed_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            packed_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            packed_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            print(f"Unsupported channel count in {file}: {img.shape}")
            continue

        print(f"  Input shape: {packed_rgb.shape}, dtype: {packed_rgb.dtype}")
        try:
            cfa = _bayer_from_packed_rgb(packed_rgb)
        except ValueError as e:
            print(f"Skipping {file}: {e}")
            continue

        print(f"  CFA shape: {cfa.shape}, dtype: {cfa.dtype}")
        _check_array_health(cfa.astype(np.float64), "CFA_input")
        
        reconstructed_rgb = _demosaic_alleysson_fourier(cfa)
        
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(output_path, f"demosaic_{name}.png")

        cv2.imwrite(save_path, cv2.cvtColor(reconstructed_rgb, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {save_path}")

    print("\nFrequency Reconstruction with Alleysson method DONE.")

