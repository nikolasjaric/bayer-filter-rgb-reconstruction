import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import windows
import matplotlib.pyplot as plt
import os
from glob import glob


def dubois_freq_demosaic(mosaic_image, method='adaptive'):
    """
    Dubois (2005) frekvencijska rekonstrukcija RGB iz Bayer RGGB mozaika.
    """
    CFA = mosaic_image.astype(np.float32) / 255.0
    H, W = CFA.shape

    m_R = np.zeros((H, W), dtype=np.float32)
    m_G = np.zeros((H, W), dtype=np.float32)
    m_B = np.zeros((H, W), dtype=np.float32)

    m_R[0::2, 1::2] = 1.0
    m_G[0::2, 0::2] = 1.0
    m_G[1::2, 1::2] = 1.0
    m_B[1::2, 0::2] = 1.0

    n1, n2 = np.meshgrid(np.arange(W), np.arange(H))
    mod_C1 = ((-1) ** (n1 + n2)).astype(np.float32)
    mod_C2_h = ((-1) ** n1).astype(np.float32)
    mod_C2_v = ((-1) ** n2).astype(np.float32)

    if method == 'asymmetric':
        filters = design_asymmetric_filters(H, W)
        H_C1 = filters['H_C1']
        H_C2_h = filters['H_C2_h']
        H_C2_v = filters['H_C2_v']

        C1 = extract_component(CFA, H_C1, mod_C1)
        C2_h = extract_component(CFA, H_C2_h, mod_C2_h)
        C2_v = extract_component(CFA, H_C2_v, mod_C2_v)
        C2 = C2_h + C2_v

    elif method == 'adaptive':
        filters = design_adaptive_filters(H, W)
        H_C1 = filters['H_C1']
        H_C2_h = filters['H_C2_h']
        H_C2_v = filters['H_C2_v']

        C1 = extract_component(CFA, H_C1, mod_C1)
        C2_h_full = extract_component(CFA, H_C2_h, mod_C2_h)
        C2_v_full = extract_component(CFA, H_C2_v, mod_C2_v)

        E_h = compute_local_energy(CFA, center_freq=(0.375, 0.0))
        E_v = compute_local_energy(CFA, center_freq=(0.0, 0.375))

        epsilon = 1e-10
        alpha = E_v / (E_h + E_v + epsilon)

        C2 = alpha * C2_h_full + (1 - alpha) * C2_v_full

    else:
        raise ValueError("Method must be 'asymmetric' or 'adaptive'")

    L = CFA - C1 * mod_C1 - C2 * (mod_C2_h + mod_C2_v)

    R = L + 0.5 * C1 + 0.5 * C2
    G = L - 0.5 * C1
    B = L + 0.5 * C1 - 0.5 * C2

    RGB = np.stack([R, G, B], axis=2)
    RGB = np.clip(RGB, 0, 1)
    RGB = (RGB * 255).astype(np.uint8)

    return RGB


def design_asymmetric_filters(H, W):
    fy, fx = np.meshgrid(
        np.fft.fftfreq(W),
        np.fft.fftfreq(H)
    )

    fc_C1 = 0.5
    sigma_C1 = 0.12

    dist_C1 = np.sqrt((fx - fc_C1)**2 + (fy - fc_C1)**2)
    H_C1 = np.exp(-(dist_C1**2) / (2 * sigma_C1**2))

    fc_h = 0.5
    sigma_h_x = 0.03
    sigma_h_y = 0.20

    H_C2_h = np.exp(-((fx - fc_h)**2) / (2 * sigma_h_x**2)
                    -((fy)**2) / (2 * sigma_h_y**2))

    fc_v = 0.5
    sigma_v_x = 0.20
    sigma_v_y = 0.03

    H_C2_v = np.exp(-((fx)**2) / (2 * sigma_v_x**2)
                    -((fy - fc_v)**2) / (2 * sigma_v_y**2))

    return {
        'H_C1': H_C1,
        'H_C2_h': H_C2_h,
        'H_C2_v': H_C2_v
    }


def design_adaptive_filters(H, W):
    fy, fx = np.meshgrid(
        np.fft.fftfreq(W),
        np.fft.fftfreq(H)
    )

    fc_C1 = 0.5
    sigma_C1 = 0.12
    dist_C1 = np.sqrt((fx - fc_C1)**2 + (fy - fc_C1)**2)
    H_C1 = np.exp(-(dist_C1**2) / (2 * sigma_C1**2))

    sigma_full = 0.15

    dist_h = np.sqrt((fx - 0.5)**2 + fy**2)
    H_C2_h = np.exp(-(dist_h**2) / (2 * sigma_full**2))

    dist_v = np.sqrt(fx**2 + (fy - 0.5)**2)
    H_C2_v = np.exp(-(dist_v**2) / (2 * sigma_full**2))

    return {
        'H_C1': H_C1,
        'H_C2_h': H_C2_h,
        'H_C2_v': H_C2_v
    }


def extract_component(CFA, H_filter, demod_func):
    F_CFA = fft2(CFA)
    F_filtered = F_CFA * H_filter
    filtered = np.real(ifft2(F_filtered))
    component = filtered * demod_func
    return component


def compute_local_energy(CFA, center_freq=(0.375, 0.0), sigma=3.5):
    H, W = CFA.shape
    fy, fx = np.meshgrid(
        np.fft.fftfreq(W),
        np.fft.fftfreq(H)
    )

    fc_x, fc_y = center_freq
    dist = np.sqrt((fx - fc_x)**2 + (fy - fc_y)**2)

    sigma_freq = 1.0 / (2 * np.pi * sigma)
    H_bandpass = np.exp(-(dist**2) / (2 * sigma_freq**2))

    F_CFA = fft2(CFA)
    filtered = np.real(ifft2(F_CFA * H_bandpass))

    energy_raw = filtered ** 2
    kernel = np.ones((5, 5)) / 25.0
    energy = cv2.filter2D(energy_raw, -1, kernel)

    return energy

"""
MAIN FUNCKIJA ZA TESTIRANJE I KORIÅ TENJE
"""

def frequency_reconstruction_dubois(input_path, output_path, method='adaptive'):
    print(f"Dubois frequency reconstruction ({method}) from: {input_path} to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    search_pattern = os.path.join(input_path, 'mosaic_*.png')
    mosaic_files = glob(search_pattern)

    if len(mosaic_files) == 0:
        print(f"Warning: No mosaic files found in {input_path}")
        return

    print(f"Found {len(mosaic_files)} mosaic images to process")

    for mosaic_file in mosaic_files:
        mosaic = cv2.imread(mosaic_file, cv2.IMREAD_GRAYSCALE)
        if mosaic is None:
            print(f"Error: Could not load {mosaic_file}")
            continue
        rgb_reconstructed = dubois_freq_demosaic(mosaic, method=method)
        base_name = os.path.basename(mosaic_file)
        new_name = base_name.replace('mosaic_', f'dubois_{method}_')
        save_path = os.path.join(output_path, new_name)
        rgb_bgr = cv2.cvtColor(rgb_reconstructed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, rgb_bgr)
        print(f"Processed: {base_name} -> {new_name}")

    print(f"Dubois reconstruction complete! Saved to {output_path}")


def visualize_filters(H=256, W=256, method='adaptive'):
    if method == 'asymmetric':
        filters = design_asymmetric_filters(H, W)
    else:
        filters = design_adaptive_filters(H, W)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(fftshift(filters['H_C1']), cmap='gray', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[0, 0].set_title('H_C1: Bandpass @ (0.5, 0.5)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].imshow(fftshift(filters['H_C2_h']), cmap='gray', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[0, 1].set_title('H_C2_h: Horizontal @ (0.5, 0)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].imshow(fftshift(filters['H_C2_v']), cmap='gray', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[1, 0].set_title('H_C2_v: Vertical @ (0, 0.5)')
    axes[1, 0].grid(True, alpha=0.3)

    combined = fftshift(filters['H_C2_h'] + filters['H_C2_v'])
    axes[1, 1].imshow(combined, cmap='gray', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[1, 1].set_title('H_C2_h + H_C2_v (combined)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'dubois_filters_{method}.png', dpi=150)
    print(f"Filter visualization saved to dubois_filters_{method}.png")
    plt.show()