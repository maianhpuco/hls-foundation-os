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
        # load first 6 bands for images, single band for labels
        if img.shape[0] > 1:  # Assuming multi-band for images
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
    print('- pd_dataframe: ', pd_dataframe)
    img_list = pd_dataframe.iloc[:, 0].tolist()
    mask_list = pd_dataframe.iloc[:, 1].tolist()
    print('-> img_list: ', img_list)
    print('-> mask_list: ', mask_list)
    
    target_substrings = [
        "India_399883",
        "Pakistan_664885",
        "Paraguay_868895"
    ]


    for idx, (cur_img_name, cur_mask_name) in enumerate(zip(filtered_img_list, filtered_mask_list)):
        print('--- {} - {} ---'.format(idx, cur_img_name))

        IMG_NAME = cur_img_name.replace('S1', 'S2')
        LABEL_NAME = cur_mask_name
        IMG_PATH = os.path.join('/project/hnguyen2/mvu9/datasets/SEN1Floods11/v1.1/data/flood_events/HandLabeled/S2Hand/', IMG_NAME)
        LABEL_PATH = os.path.join('/project/hnguyen2/mvu9/datasets/SEN1Floods11/v1.1/data/flood_events/HandLabeled/LabelHand', LABEL_NAME)

        # Grab the config and model weights
        config_path = hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename="sen1floods11_Prithvi_100M.py")
        ckpt = '/project/hnguyen2/hqvo3/courseworks/codes/prithvi/checkpoints/init_exp_vanilla/best_mIoU_epoch_90.pth'
        
        print('[**] type(ckpt): ', type(ckpt))
        print('[**] ckpt: ', ckpt)
        
        finetuned_model = init_segmentor(Config.fromfile(config_path), ckpt, device="cpu")
    
        # Load and process input image
        input_data_inference = load_raster(IMG_PATH)
        print(f"Image input shape is {input_data_inference.shape}")
        raster_for_visualization = enhance_raster_for_visualization(input_data_inference)
        
        # Load and process label (ground truth)
        label_data_inference = load_raster(LABEL_PATH)  # Single-band raster
        print('[!!] label_data_inference shape: ', label_data_inference.shape)
        # Squeeze to remove channel dimension (1, H, W) -> (H, W)
        label_data_inference = np.squeeze(label_data_inference)
        
        # Run inference
        custom_test_pipeline = process_test_pipeline(finetuned_model.cfg.data.test.pipeline)
        result = inference_segmentor(finetuned_model, IMG_PATH, custom_test_pipeline=custom_test_pipeline)
    
        # Plotting
        fig, ax = plt.subplots(1, 4, figsize=(15, 10))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
        
        ax[0].imshow(raster_for_visualization)
        ax[0].set_title("Input Image", fontsize=12)
        
        # Plot ground truth directly as single-band integer labels
        ax[1].imshow(label_data_inference, norm=norm, cmap="jet")
        ax[1].set_title("Ground Truth", fontsize=12)
        
        ax[2].imshow(result[0], norm=norm, cmap="jet")
        ax[2].set_title("Full Fine-tuning", fontsize=12)
        
        ax[3].imshow(raster_for_visualization)
        ax[3].imshow(result[0], cmap="jet", alpha=0.3, norm=norm)
        ax[3].set_title("Overlay", fontsize=12)
        
        for subplot in ax:
            subplot.axis('off')
        
        # Save instead of display
        os.makedirs('./plotting_2/full', exist_ok=True)
        plt.savefig('./plotting_2/full/vis_{}.png'.format(IMG_NAME.rsplit('.', 1)[0]), bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory