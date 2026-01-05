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
    Note: Despite the function name (_rggb_masks), this project uses GRBG pattern.
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
    Pretvori 'packed' 3-kanalni Bayer mozaik (R/G/B samo na njihovim pozicijama, ostalo 0)
    u jedan-kanalni Bayer CFA signal (skalar po pikselu).
    """
    if packed_rgb.ndim != 3 or packed_rgb.shape[2] < 3:
        raise ValueError("packed_rgb must be HxWx3 (or HxWx4) array")

    # uzimamo samo prva 3 kanala (RGB)
    rgb = packed_rgb[..., :3]
    H, W, _ = rgb.shape
    bayer = np.zeros((H, W), dtype=rgb.dtype)

    R_mask, G_mask, B_mask = _rggb_masks((H, W))

    # uzmi vrijednosti iz odgovarajućih kanala na pozicijama maske
    bayer[R_mask] = rgb[..., 0][R_mask]
    bayer[G_mask] = rgb[..., 1][G_mask]
    bayer[B_mask] = rgb[..., 2][B_mask]

    return bayer

def _estimate_luminance_fourier(cfa: np.ndarray, cutoff: float = 0.45) -> np.ndarray:
    """ Procjena luminancije iz CFA slike putem niskopropusnog filtera u Fourierovoj domeni.
    
    A–S–H: Luminancija je niskofrekventna komponenta CFA,
    dobivena separacijom frekvencija. Cutoff prilagođen za Bayerovu sliku.
    cutoff: radijalni cut-off u normaliziranoj frekvencijskoj domeni (0–0.5). 
    
    WICHTIG: cutoff=0.45 je preporučeno od Alleysson papira za GRBG/RGGB Bayer.
    Manja vrijednost (npr. 0.35) uklanja previše boje-signala.
    """
    
    cfa_f = cfa.astype(np.float64)
    H, W = cfa_f.shape
    
    # Primijeni periodičke granične uvjete (FFT pretpostavka) izbjegavanjem edge artefakata
    # Pad s refleksijom prije FFT-a
    cfa_padded = cv2.copyMakeBorder(cfa_f, H//4, H//4, W//4, W//4, cv2.BORDER_REFLECT)
    H_pad, W_pad = cfa_padded.shape
    
    F = np.fft.fft2(cfa_padded)
    F = np.fft.fftshift(F)  # Pomakni DC komponentu u centar
    
    # Normalizirane frekvencije
    u = np.fft.fftfreq(W_pad)
    v = np.fft.fftfreq(H_pad)
    u = np.fft.fftshift(u)
    v = np.fft.fftshift(v)
    U, V = np.meshgrid(u, v)
    R = np.sqrt(U**2 + V**2)
    
    # Gaussov niskopropusni filtar: sigma prilagođen cutoff-u
    sigma = cutoff / np.sqrt(2 * np.log(2))
    H_low = np.exp(-(R**2) / (2 * sigma**2))
    
    F_lum = F * H_low
    F_lum = np.fft.ifftshift(F_lum)  # Pomakni prije inverzne transformacije
    lum_padded = np.fft.ifft2(F_lum).real
    
    # Ukloni padding
    lum = lum_padded[H//4:H//4+H, W//4:W//4+W]
    
    # FIX #2: Normalize luminance to input CFA range (0-255)
    # FFT is unscaled; ensure output matches input scale for proper chrominance extraction
    lum_abs_max = np.abs(lum).max()
    if lum_abs_max > 0:
        # Preserve relative magnitudes while scaling to CFA input range
        cfa_max = np.abs(cfa_f).max()
        lum = lum * (cfa_max / lum_abs_max)
    
    return lum.astype(np.float64)

def _interpolate_chrominance(ch: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Interpolacija krominancije: Gaussovo zamućenje s BORDER_REFLECT za konzistentnost
    s Alleysson metodom (izbjegavanje edge diskontinuiteta).
    
    ch: poduzorkani krominacijski kanal (samo na R/G/B pozicijama, ostalo 0).
    """
    ch_f = ch.astype(np.float64)
    if ksize % 2 == 0:
        ksize += 1
    # Koristi BORDER_REFLECT umjesto BORDER_DEFAULT radi interpolacijskog integriteta
    ch_interp = cv2.GaussianBlur(
        ch_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )
    return ch_interp

def _check_array_health(arr: np.ndarray, name: str = "array") -> None:
    """Diagnostic: Check for NaN, Inf, dtype, and value range."""
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    min_val = np.nanmin(arr) if arr.size > 0 else None
    max_val = np.nanmax(arr) if arr.size > 0 else None
    mean_val = np.nanmean(arr) if arr.size > 0 else None
    
    print(f"  [{name}] dtype={arr.dtype}, shape={arr.shape}, NaN={nan_count}, Inf={inf_count}, "
          f"min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
    
    if nan_count > 0 or inf_count > 0:
        print(f"    WARNING: {name} contains NaN/Inf values; clamping...")
        arr[~np.isfinite(arr)] = 0.0

def _demosaic_alleysson_fourier(cfa: np.ndarray) -> np.ndarray:
    """
    Alleysson–Süsstrunk–Hérault demosaicing kernela:
    1) Luminancija Y = niskopropusno filtriran CFA (frekvencijska separacija).
    2) Krominancija C = CFA - Y (visokofrekventna komponenta).
    3) Demultipleksiranje C po GRBG maskama na tri krominacijska kanala.
    4) Interpolacija svakog krominacijskog kanala (Gaussovo zamućenje).
    5) Rekonstrukcija: R=Y+C_R, G=Y+C_G, B=Y+C_B (aditivni model).
    
    Ključna svojstva:
    - Luminancija čuva niskofrekventnu strukturu i sprječava aliasing.
    - Krominancija je izdvojena kao ostatak, interpolirana sa manjom rezolucijom.
    - Aditivnost omogućava nezavisnu obradu L i C kanala.
    """
    H, W = cfa.shape
    
    # 1) Luminancija: Fourier-bazirana frekvencijska separacija
    print("  [Luminance] Computing Fourier-based low-pass filter...")
    Y = _estimate_luminance_fourier(cfa, cutoff=0.45)  # FIX #3: Increased from 0.35 to 0.45
    _check_array_health(Y, "Luminance Y")
    
    # 2) Krominancija kao ostatak
    print("  [Chrominance] Extracting residual C = CFA - Y...")
    C_scalar = cfa.astype(np.float64) - Y
    _check_array_health(C_scalar, "Residual C")
    
    # 3) Demultipleksiranje krominancije po GRBG pozicijama
    print("  [Demultiplex] Separating R, G, B chrominance channels...")
    R_mask, G_mask, B_mask = _rggb_masks((H, W))
    
    C_R = np.zeros((H, W), dtype=np.float64)
    C_G = np.zeros((H, W), dtype=np.float64)
    C_B = np.zeros((H, W), dtype=np.float64)
    
    C_R[R_mask] = C_scalar[R_mask]
    C_G[G_mask] = C_scalar[G_mask]
    C_B[B_mask] = C_scalar[B_mask]
    
    _check_array_health(C_R, "C_R")
    _check_array_health(C_G, "C_G")
    _check_array_health(C_B, "C_B")
    
    # 4) Interpolacija krominancije (Gaussov blur s refleksivnom granicom)
    print("  [Interpolate] Gaussian blur on chrominance channels...")
    C_R_full = _interpolate_chrominance(C_R, ksize=5, sigma=1.0)
    C_G_full = _interpolate_chrominance(C_G, ksize=5, sigma=1.0)
    C_B_full = _interpolate_chrominance(C_B, ksize=5, sigma=1.0)
    
    _check_array_health(C_R_full, "C_R_full")
    _check_array_health(C_G_full, "C_G_full")
    _check_array_health(C_B_full, "C_B_full")
    
    # 5) Rekonstrukcija RGB-a: aditivna kombinacija luminancije i interpolirane krominancije
    print("  [Reconstruct] RGB = Y + C channels...")
    R = Y + C_R_full
    G = Y + C_G_full
    B = Y + C_B_full
    
    _check_array_health(R, "R_raw")
    _check_array_health(G, "G_raw")
    _check_array_health(B, "B_raw")
    
    # Normalizacija: preslikaj iz [0, 255] float u [0, 255] uint8
    # Ako su ulazne vrijednosti u razinu 0–255, nakon obrade trebaju biti klampirane
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)
    
    rgb = np.stack([R, G, B], axis=-1)
    rgb = rgb.astype(np.uint8)
    
    _check_array_health(rgb.astype(np.float64), "RGB_final")
    
    return rgb

def frequency_reconstruction_alleysson(input_path: str, output_path: str) -> None:
    """
    Glavna funkcija koju GUI ocekuje.
    input_path: folder s .png Bayer mozaicima (3-kanalni 'packed').
    output_path: folder za rekonstruirane RGB slike (demosaic_<ime>.png).
    """
    os.makedirs(output_path, exist_ok=True)
    print(f"Running Frequency Reconstruction (Fourier) demosaicing from: {input_path} to: {output_path}")

    mosaic_files = glob(os.path.join(input_path, "*.png"))
    if not mosaic_files:
        print("No .png mosaic files found in input_path.")
        return

    for file in mosaic_files:
        print(f"\nProcessing: {file}")
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not read {file}")
            continue

        # Podrzi 3-kanalni i 4-kanalni ulaz; ako je grayscale konvertiraj u RGB
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