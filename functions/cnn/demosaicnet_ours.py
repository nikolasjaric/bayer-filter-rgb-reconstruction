#!/usr/bin/env python
# MIT License
#
# Deep Joint Demosaicking and Denoising (prilagođeno za pre-mosaicked ulaz)
# Samo za Bayer mozaik (bez X-Trans), batch obrada nad folderom ulaza.

import argparse
import skimage.io
import numpy as np
import time
import torch as th
from tqdm import tqdm
from pathlib import Path
from demosaic import modules, converter

NOISE_LEVELS = [0.0000, 0.0785]  # min/max šuma za koji je mreža trenirana
SUPPORTED_EXTS = {".png", ".jpg"}


def _uint2float(I):
    if I.dtype == np.uint8:
        I = I.astype(np.float32) * (1.0 / 256.0)
    elif I.dtype == np.uint16:
        I = I.astype(np.float32) / 65535.0
    elif I.dtype in [np.float16, np.float32, np.float64]:
        I = I.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {I.dtype}")
    return I


def _float2uint(I, dtype):
    if dtype == np.uint8:
        I = I / (1.0 / 256.0)
        I += 0.5
        I = np.clip(I, 0, 255).astype(np.uint8)
    elif dtype == np.uint16:
        I = I * 65535.0
        I += 0.5
        I = np.clip(I, 0, 65535).astype(np.uint16)
    else:
        # ako je ulaz bio float, spremi kao uint8
        I = np.clip(I * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return I


def demosaick(net, M, noiselevel, tile_size, crop):
    dev = next(net.parameters()).device
    M = th.from_numpy(M).to(device=dev, dtype=th.float)

    _, _, h, w = M.shape
    out_ref = th.zeros(3, h, w, device=dev, dtype=th.float)
    sigma_noise = noiselevel * th.ones(1, device=dev, dtype=th.float)

    tile_size = min(min(tile_size, h), w)

    mod = 2  # Bayer uzorak ima period 2
    tile_step = tile_size - crop * 2
    tile_step = tile_step - (tile_step % mod)

    start = time.time()
    for start_x in range(0, w, tile_step):
        end_x = start_x + tile_size
        if end_x > w:
            end_x = w
            start_x = end_x - tile_size
            start_x = start_x - (start_x % mod)
            end_x = start_x + tile_size
        for start_y in range(0, h, tile_step):
            end_y = start_y + tile_size
            if end_y > h:
                end_y = h
                start_y = end_y - tile_size
                start_y = start_y - (start_y % mod)
                end_y = start_y + tile_size

            sample = {
                "mosaic": M[:, :, start_y:end_y, start_x:end_x],
                "noise_level": sigma_noise,
            }

            outr = net(sample)  # (B,3,oh,ow)
            oh, ow = outr.shape[2:]
            ch = (tile_size - oh) // 2
            cw = (tile_size - ow) // 2
            out_ref[
                :,
                start_y + ch : start_y + ch + oh,
                start_x + cw : start_x + cw + ow,
            ] = outr[0]

    # print("running time:", time.time() - start)
    return out_ref.detach().cpu().numpy(), 0.0


def demosaick_load_model(network_path, noiselevel=0.0):
    print("Loading Caffe weights")
    print("Noise level:", noiselevel)
    noiselevel = float(noiselevel)
    # Ako je zadana razina šuma različita od 0, koristi model treniran za šum
    if noiselevel != 0.0: network_path = "pretrained_models\\bayer_noise"

    if noiselevel == 0.0:
        print("Using noise-free model")
        model_ref = modules.get({"model": "BayerNetwork"})
        cvt = converter.Converter(network_path, "BayerNetwork")
    else:
        model_ref = modules.get({"model": "BayerNetworkNoise"})
        cvt = converter.Converter(network_path, "BayerNetworkNoise")

    cvt.convert(model_ref)
    for p in model_ref.parameters():
        p.requires_grad = False
    return model_ref


def process_one_image(model_ref, img_path: Path, out_dir: Path, noise: float, tile_size: int, crop: int):
    # Učitaj pre-mosaicked sliku (H,W,3)
    inp = skimage.io.imread(str(img_path))

    # Makni alpha ako postoji
    if inp.ndim == 3 and inp.shape[2] == 4:
        inp = inp[:, :, :3]

    if inp.ndim != 3 or inp.shape[2] != 3:
        raise ValueError(f"{img_path.name}: expected a pre-mosaicked 3-channel image (H,W,3), got shape {inp.shape}")

    in_dtype = inp.dtype
    M = _uint2float(inp)

    # (H,W,3) -> (1,3,H,W)
    M4 = M.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    # Padding (simetričan)
    c = crop + (crop % 2)
    M4 = np.pad(M4, [(0, 0), (0, 0), (c, c), (c, c)], mode="symmetric")

    # Inference
    R, _ = demosaick(model_ref, M4, noise, tile_size, crop)

    # Ukloni padding i preoblikuj u (H,W,3)
    R = R.squeeze().transpose(1, 2, 0)
    R = R[c:-c, c:-c, :]

    # Spremi samo demosaicked rezultat
    base = img_path.stem
    out_img = out_dir / f"{base}_demosaic.png"

    out = _float2uint(R, in_dtype if in_dtype in [np.uint8, np.uint16] else np.uint8)
    skimage.io.imsave(str(out_img), out)



def main(args):
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skupi sve podržane slike (ne rekuzivno)
    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()])
    if not images:
        raise SystemExit(f"Nema slika u {in_dir} s ekstenzijama: {sorted(SUPPORTED_EXTS)}")

    # Učitaj mrežu jednom
    model_ref = demosaick_load_model(args.net_path, args.noise)
    if args.gpu:
        model_ref.cuda()
    else:
        model_ref.cpu()

    crop = 48
    print(f"Crop {crop}")
    print(f"Found {len(images)} images. Processing...")

    errors = []
    for img_path in tqdm(images, unit="img"):
        try:
            process_one_image(model_ref, img_path, out_dir, args.noise, args.tile_size, crop)
        except Exception as e:
            errors.append((img_path.name, str(e)))

    if errors:
        print("\nZavršeno s greškama na sljedećim datotekama:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    else:
        print("\nSve slike uspješno obrađene.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the demosaicking network on a folder of pre-mosaicked 3-channel Bayer images."
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="path to folder with pre-mosaicked 3-channel images (H,W,3).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="path to folder where outputs will be saved.")
    parser.add_argument("--net_path", type=str, default="pretrained_models\\bayer",
                        help="path to model folder with .npy weights.")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Std-dev of additive Gaussian noise (for noise-trained model selection).")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size for sliding-window inference.")
    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="Use GPU if available.")
    parser.set_defaults(gpu=False)

    args = parser.parse_args()

    if not (NOISE_LEVELS[0] <= args.noise <= NOISE_LEVELS[1]):
        raise ValueError(f"The model was trained on noise levels in [{NOISE_LEVELS[0]}, {NOISE_LEVELS[1]}]")

    main(args)

'''primjer pokretanja:
   python demosaicnet_ours.py --input_dir "C:/path/do/ulaza" --output_dir "C:/path/do/izlaza" --noise 0.0
   
'''

