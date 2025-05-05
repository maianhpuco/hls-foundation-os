import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import rasterio
import yaml

from mmcv import Config
from mmseg.models import build_segmentor
from mmseg.datasets.pipelines import Compose, LoadImageFromFile
from mmseg.apis import init_segmentor
from model_inference import inference_segmentor, process_test_pipeline
from huggingface_hub import hf_hub_download
import matplotlib
from torch import nn


NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

def load_raster(path, crop=None):
    with rasterio.open(path) as src:
        img = src.read()

        # load first 6 bands
        img = img[:6]

        img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
        if crop:
            img = img[:, -crop[0]:, -crop[1]:]
    return img

def enhance_raster_for_visualization(raster, ref_img=None):
    if ref_img is None:
        ref_img = raster
    channels = []
    for channel in range(raster.shape[0]):
        valid_mask = np.ones_like(ref_img[channel], dtype=bool)
        valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
        mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
        normalized_raster = (raster[channel] - mins) / (maxs - mins)
        normalized_raster[~valid_mask] = 0
        clipped = np.clip(normalized_raster, 0, 1)
        channels.append(clipped)
    clipped = np.stack(channels)
    channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
    rgb = channels_last[..., ::-1]
    return rgb

if __name__ == '__main__':
    SPLIT_CSV_PATH = '/project/hnguyen2/mvu9/datasets/SEN1Floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv'
    pd_dataframe = pd.read_csv(SPLIT_CSV_PATH)
    img_list = pd_dataframe.iloc[:, 0].tolist()
    mask_list = pd_dataframe.iloc[:, 1].tolist()
    REPO_ROOT_DIR = '/project/hnguyen2/mvu9/folder_04_ma/hls-foundation-os'
    # Define model variants to compare
    ckpt_variants = {
        "prompt08": f"{REPO_ROOT_DIR}/nprompt_08/latest.pth",
        "prompt16": f"{REPO_ROOT_DIR}/nprompt_16/latest.pth",
        "prompt32": f"{REPO_ROOT_DIR}/nprompt_32/latest.pth",
    }

    for idx, cur_img_name in enumerate(img_list):
        print(f'--- {idx} - {cur_img_name} ---')
        cur_mask_name = mask_list[idx]
        IMG_NAME = cur_img_name.replace('S1', 'S2')
        LABEL_NAME = cur_mask_name

        IMG_PATH = os.path.join('/project/hnguyen2/mvu9/datasets/SEN1Floods11/v1.1/data/flood_events/HandLabeled/S2Hand/', IMG_NAME)
        LABEL_PATH = os.path.join('/project/hnguyen2/mvu9/datasets/SEN1Floods11/v1.1/data/flood_events/HandLabeled/LabelHand', LABEL_NAME)

        config_path = hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename="sen1floods11_Prithvi_100M.py")

        input_data_inference = load_raster(IMG_PATH)
        raster_for_visualization = enhance_raster_for_visualization(input_data_inference)
        label_data_inference = load_raster(LABEL_PATH)
        label_data_inference = np.transpose(label_data_inference, (1, 2, 0))

        # Visualization
        fig, ax = plt.subplots(1, len(ckpt_variants) + 2, figsize=(20, 8))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2)

        ax[0].imshow(raster_for_visualization)
        ax[0].set_title("RGB Input")
        ax[1].imshow(label_data_inference, norm=norm, cmap="jet")
        ax[1].set_title("Ground Truth")

        # Loop over checkpoints and get predictions
        for i, (label, ckpt_path) in enumerate(ckpt_variants.items()):
            print(f"[{label}] Loading checkpoint: {ckpt_path}")
            model = init_segmentor(Config.fromfile(config_path), ckpt_path, device="cpu")
            custom_test_pipeline = process_test_pipeline(model.cfg.data.test.pipeline)
            pred = inference_segmentor(model, IMG_PATH, custom_test_pipeline=custom_test_pipeline)

            ax[i + 2].imshow(pred[0], norm=norm, cmap="jet")
            ax[i + 2].set_title(f"Pred: {label}")

        for subplot in ax:
            subplot.axis('off')

        # Save output
        os.makedirs('./vis/outputs/compare_prompts', exist_ok=True)
        save_name = IMG_NAME.rsplit('.', 1)[0]
        plt.savefig(f'./vis/outputs/compare_prompts/vis_{save_name}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
