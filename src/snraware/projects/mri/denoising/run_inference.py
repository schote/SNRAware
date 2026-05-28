"""Run MRI inference for denoising."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from colorama import Fore, Style
from omegaconf import OmegaConf

from snraware.components.setup import end_timer, start_timer
from snraware.projects.mri.denoising.inference import apply_model
from snraware.projects.mri.denoising.inference_model import (
    load_lit_model,
    load_model,
    load_scripted_model,
)

# -------------------------------------------------------------------------------------------------

def load_input(input_dir, input_fname):
    input_dir = Path(input_dir)
    if input_fname.endswith(".npy"):
        image = np.load(input_dir / input_fname)
    else:
        image = (
            np.load(input_dir / f"{input_fname}_real.npy") + 1j * np.load(input_dir / f"{input_fname}_imag.npy")
        )
    # Normalize input image to [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def run_inference(
    saved_model_path,
    saved_config_path,
    input_dir,
    input_fname,
    gmap_fname,
    no_gmap,
    scaling_factor=1.0,
    batch_size=1,
    cutout=None,
    overlap=(16, 16, 8),
    device=None,
):
    """Run inference on the input data using the specified model and configuration."""
    # load the model
    if saved_model_path.endswith(".pts"):
        model = load_scripted_model(saved_model_path)
        config = OmegaConf.load(saved_config_path)
    elif saved_model_path.endswith(".pth") or saved_model_path.endswith(".ckpt"):
        model, config = load_model(saved_model_path, saved_config_path)
    else:
        model, config = load_lit_model(saved_model_path, saved_config_path)
    print(
        f"{Fore.YELLOW}Load in model file - {saved_model_path} - {saved_config_path}{Style.RESET_ALL}"
    )

    if saved_model_path.endswith(".ckpt"):
        from snraware.projects.mri.denoising.lightning_denoising import after_training

        config.logging.output_dir = os.path.dirname(saved_model_path)
        after_training(model, config)

    # -------------------------------------------
    # load the data
    image = load_input(input_dir, input_fname)

    image = np.squeeze(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    if no_gmap:
        gmap = np.ones([image.shape[0], image.shape[1], image.shape[2]])
    else:
        gmap_file = f"{input_dir}/{gmap_fname}.npy"
        if os.path.exists(gmap_file):
            gmap = np.load(f"{input_dir}/{gmap_fname}.npy")

    print(f"{Fore.YELLOW}Load in data - {image.shape} - gmap {gmap.shape}{Style.RESET_ALL}")

    batch_size = batch_size
    cutout = tuple(config.dataset.cutout_shape)
    if cutout is not None:
        cutout = tuple(cutout)

    t0 = start_timer(enable=True)
    output = apply_model(
        model=model,
        data=image,
        gmap=gmap,
        scaling_factor=scaling_factor,
        cutout=cutout,
        overlap=tuple(overlap),
        batch_size=batch_size,
        device=device,
        verbose=True,
    )
    print(
        f"{Fore.YELLOW}Inference time {end_timer(t=t0, enable=True, verbose=False) / 1e3:.2f} sec{Style.RESET_ALL}"
    )
    return output, image


# -------------------------------------------------------------------------------------------------


def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup.
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT MRI test evaluation")

    parser.add_argument("--input_dir", default=None, help="folder to load the data")
    parser.add_argument("--output_dir", default=None, help="folder to save the data")
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=1.0,
        help="scaling factor to adjust model strength; higher scaling means lower strength",
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default=None,
        help='model path. endswith ".pth" or ".pts" or ".ckpt"',
    )
    parser.add_argument(
        "--saved_config_path", type=str, default=None, help="model config yaml file."
    )

    parser.add_argument("--batch_size", type=int, default=1, help="batch size for inference")
    parser.add_argument(
        "--cutout",
        nargs="+",
        type=int,
        default=None,
        help="cutout size for (H, W, T), None means using the config setting",
    )
    parser.add_argument(
        "--overlap",
        nargs="+",
        type=int,
        default=[16, 16, 8],
        help="overlap for (H, W, T), (0, 0, 0) means no overlap",
    )

    parser.add_argument("--input_fname", type=str, default="input", help="input file name")
    parser.add_argument("--gmap_fname", type=str, default="gmap", help="gmap input file name")

    parser.add_argument(
        "--no_gmap", action="store_true", help="if set, do not load gmap, but set gmap as all ones"
    )

    parser.add_argument(
        "--show", action="store_true", help="if set, show input and output images"
    )

    return parser.parse_args()


# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results


def main():
    args = arg_parser()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"{Fore.YELLOW}Find devices - {device}{Style.RESET_ALL}")

    assert os.path.exists(args.input_dir), f"input_dir {args.input_dir} does not exist"
    assert os.path.exists(args.saved_model_path), (
        f"saved_model_path {args.saved_model_path} does not exist"
    )
    assert os.path.exists(args.saved_config_path), (
        f"saved_config_path {args.saved_config_path} does not exist"
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"{Fore.YELLOW}Create output_dir {args.output_dir}{Style.RESET_ALL}")

    output_image, input_image = run_inference(
        args.saved_model_path,
        args.saved_config_path,
        args.input_dir,
        args.input_fname,
        args.gmap_fname,
        args.no_gmap,
        scaling_factor=args.scaling_factor,
        batch_size=args.batch_size,
        cutout=args.cutout,
        overlap=tuple(args.overlap),
        device=device,
    )

    output_fname = args.input_fname.split(".")[0] + "_output.npy"
    input_fname = args.input_fname.split(".")[0] + "_input.npy"
    result_dir = Path(args.output_dir)

    print(f"Saving result to {result_dir} with filename {output_fname}, output - {output_image.shape}")
    # Save complex data
    np.save(result_dir / output_fname, output_image)
    # Save real/imag data
    # np.save(os.path.join(args.output_dir, "output_real.npy"), output.real)
    # np.save(os.path.join(args.output_dir, "output_imag.npy"), output.imag)
    # Save input image
    np.save(result_dir / input_fname, input_image)

    if args.show:
        import matplotlib.pyplot as plt

        original = load_input(args.input_dir, args.input_fname)
        original_mag = np.squeeze(np.abs(original))
        original_mag = (original_mag - np.min(original_mag)) / (np.max(original_mag) - np.min(original_mag))

        output_mag = np.squeeze(np.abs(output))
        output_mag = (output_mag - np.min(output_mag)) / (np.max(output_mag) - np.min(output_mag))

        k = np.argmax(np.sum(original_mag, axis=(-2, -1)))

        fig, ax = plt.subplots(1, 2)
        vmin = 0
        vmax = np.max(original_mag[k])
        ax[0].imshow(original_mag[k], cmap="gray", vmin=vmin, vmax=vmax)
        ax[1].imshow(output_mag[k], cmap="gray", vmin=vmin, vmax=vmax)
        plt.show()

if __name__ == "__main__":
    main()
