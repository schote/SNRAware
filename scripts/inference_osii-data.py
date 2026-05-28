# %%
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
from colorama import Fore, Style
from omegaconf import OmegaConf
from snraware.components.setup import end_timer, start_timer
from snraware.projects.mri.denoising.inference import apply_model
from snraware.projects.mri.denoising.inference_model import load_scripted_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# %%
# Configuration
MODEL_SIZE = "medium"
MODEL_PATH = f"/home/schote01/code/SNRAware/{MODEL_SIZE}/snraware_{MODEL_SIZE}_model.pts"
MODEL_CONFIG_PATH = f"/home/schote01/code/SNRAware/{MODEL_SIZE}/snraware_{MODEL_SIZE}_model.yaml"
INPUT_DIR = "/home/schote01/data/low-field-in-vivo-data"
# INPUT_FILE = "pd_invivo_3d_zyx.npy"
INPUT_FILE = "t2_invivo_3d_zyx.npy"
OUTPUT_DIR = f"/home/schote01/data/snr-aware-denoising/{MODEL_SIZE}"

# %%
model = load_scripted_model(MODEL_PATH)
config = OmegaConf.load(MODEL_CONFIG_PATH)

# load the data
input_dir = Path(INPUT_DIR)
image = np.load(input_dir / INPUT_FILE)
image = np.squeeze(image)

image = np.transpose(image, (2, 1, 0))

# Normalize image to [0, 1]
image_mag = np.abs(image)
image = (image - np.min(image_mag)) / (np.max(image_mag) - np.min(image_mag))

# No gmap provided, use all ones
gmap = np.ones([image.shape[0], image.shape[1], image.shape[2]])
# gmap_file = f"{input_dir}/{gmap_fname}.npy"
# if os.path.exists(gmap_file):
#     gmap = np.load(f"{input_dir}/{gmap_fname}.npy")

print(f"{Fore.YELLOW}Load in data - {image.shape} - gmap {gmap.shape}{Style.RESET_ALL}")

batch_size = 1
cutout = tuple(config.dataset.cutout_shape)
if cutout is not None:
    cutout = tuple(cutout)

# %%
t0 = start_timer(enable=True)
image_denoised = apply_model(
    model=model,
    data=image,
    gmap=gmap,
    # scaling_factor=1.0,
    scaling_factor=20.0,
    cutout=cutout,
    overlap=(20, 20, 1),
    # overlap=(16, 16, 8),
    # overlap=(0, 0, 0),
    batch_size=batch_size,
    device=device,
    verbose=True,
)
print(
    f"{Fore.YELLOW}Inference time {end_timer(t=t0, enable=True, verbose=False) / 1e3:.2f} sec{Style.RESET_ALL}"
)

# %%
image_mag = np.abs(image)
image_mag = image_mag / np.max(image_mag)
image_denoised_mag = np.abs(image_denoised)
image_denoised_mag = image_denoised_mag / np.max(image_denoised_mag)

# k = np.argmax(np.sum(image, axis=(0, 1)))
vmin = 0
vmax = 0.3    #np.max(np.abs(image[..., k]))

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(6, 3), layout="tight")
ax[0].imshow(image_mag[..., 9], cmap="gray", vmin=vmin, vmax=vmax)
ax[1].imshow(image_denoised_mag[..., 9], cmap="gray", vmin=vmin, vmax=vmax)
fig.show()

# for k in range(image_mag.shape[-1]):
#     fig, ax = plt.subplots(1, 2, sharey=True, figsize=(6, 3), layout="tight")
#     ax[0].imshow(image_mag[..., k], cmap="gray", vmin=vmin, vmax=vmax)
#     ax[1].imshow(image_denoised_mag[..., k], cmap="gray", vmin=vmin, vmax=vmax)
#     fig.show()


# %%
result_dir = Path(OUTPUT_DIR)
output_fname = INPUT_FILE.split(".")[0] + "_output.npy"
input_fname = INPUT_FILE.split(".")[0] + "_input.npy"

# print(f"Saving result to {result_dir} with filename {output_fname}, output - {image_denoised.shape}")
# np.save(result_dir / output_fname, image_denoised)
# np.save(result_dir / input_fname, image)


