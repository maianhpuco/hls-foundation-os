import os
import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from mmcv import Config
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose
from mmseg.datasets.builder import PIPELINES

warnings.filterwarnings("ignore")

# Custom pipeline transform to load images with rasterio
@PIPELINES.register_module()
class LoadImageWithRasterio:
    def __init__(self, to_float32=False, nodata=None, nodata_replace=0):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace

    def __call__(self, results):
        filename = results["img_info"]["filename"]
        try:
            with rasterio.open(filename) as src:
                img = src.read()  # Shape: (C, H, W)
        except Exception as e:
            raise Exception(f"Failed to open {filename}: {e}")
        if self.nodata is not None:
            img = np.where(img == self.nodata, self.nodata_replace, img)
        if self.to_float32:
            img = img.astype(np.float32)
        results["img"] = img
        results["img_shape"] = img.shape[1:]
        results["ori_shape"] = img.shape[1:]
        results["filename"] = filename
        results["ori_filename"] = os.path.basename(filename)
        return results

# Custom inference function to handle preprocessed dictionary
def custom_inference_segmentor(model, data):
    model.eval()
    with torch.no_grad():
        img = data["img"].unsqueeze(0)  # [1, C, H, W]
        img_metas = [data["img_metas"]]
        if torch.cuda.is_available():
            img = img.cuda()
        result = model(return_loss=False, img=img, img_metas=img_metas)
    return result

# Visualization enhancement
NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

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
    return channels_last[..., ::-1]  # RGB

# Paths and settings
config_path = "configs/sen1floods11_config_prompt_tuning_16.py"
checkpoint_path = "/project/hnguyen2/mvu9/folder_04_ma/hls-foundation-os/nprompt_16/best_mIoU_epoch_80.pth"
output_dir = "./vis/outputs/nprompt_16"
os.makedirs(output_dir, exist_ok=True)
split_csv_path = "/project/hnguyen2/mvu9/datasets/SEN1Floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv"

# Load config
cfg = Config.fromfile(config_path)
model = init_segmentor(cfg, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")

# Dataset params
img_dir = cfg.img_dir
ann_dir = cfg.ann_dir
bands = cfg.bands
rgb_bands = [bands.index(i) for i in [1, 2, 3]]
ignore_index = cfg.ignore_index

# Compose test pipeline
test_pipeline = Compose([
    dict(type="LoadImageWithRasterio", to_float32=False, nodata=cfg.image_nodata, nodata_replace=cfg.image_nodata_replace),
    dict(type="BandsExtract", bands=cfg.bands),
    dict(type="ConstantMultiply", constant=cfg.constant),
    dict(type="ToTensor", keys=["img"]),
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **cfg.img_norm_cfg),
    dict(type="Reshape", keys=["img"], new_shape=(len(cfg.bands), cfg.num_frames, -1, -1), look_up={"2": 1, "3": 2}),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(type="CollectTestList", keys=["img"], meta_keys=["img_info", "filename", "ori_filename", "img_shape", "ori_shape", "img_norm_cfg"]),
])

# Load split
df = pd.read_csv(split_csv_path)
img_list = df.iloc[:, 0].tolist()
mask_list = df.iloc[:, 1].tolist()

palette = {0: [0, 0, 255], 1: [255, 0, 0], 2: [128, 128, 128]}

def apply_colormap(mask, palette):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        colored[mask == cls] = color
    return colored

# Run loop
for idx, img_name in enumerate(img_list):
    print(f"--- {idx} - {img_name} ---")
    mask_name = mask_list[idx]
    img_name = img_name.replace("S1", "S2")
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(ann_dir, mask_name)

    data = {"img_info": {"filename": img_path}}
    try:
        data = test_pipeline(data)
        if isinstance(data["img"], list):
            print(f"Image is a list of {len(data['img'])} tensors.")
            data["img"] = data["img"][0]
        else:
            print(f"Image shape: {data['img'].shape}")
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        continue

    try:
        result = custom_inference_segmentor(model, data)
        pred_mask = result[0]
    except Exception as e:
        print(f"Error inferring {img_name}: {e}")
        continue

    try:
        with rasterio.open(img_path) as src:
            input_data = src.read()
        input_data = np.where(input_data == cfg.image_nodata, cfg.image_nodata_replace, input_data)
        raster_vis = enhance_raster_for_visualization(input_data[rgb_bands])
    except Exception as e:
        print(f"Error loading image {img_name}: {e}")
        continue

    try:
        with rasterio.open(label_path) as src:
            label_data = src.read(1)
    except Exception as e:
        print(f"Error loading label {mask_name}: {e}")
        continue

    gt_colored = apply_colormap(label_data, palette)
    pred_colored = apply_colormap(pred_mask, palette)

    fig, ax = plt.subplots(1, 4, figsize=(15, 10))
    ax[0].imshow(raster_vis); ax[0].set_title("Input RGB"); ax[0].axis("off")
    ax[1].imshow(gt_colored); ax[1].set_title("Ground Truth"); ax[1].axis("off")
    ax[2].imshow(pred_colored); ax[2].set_title("Prediction"); ax[2].axis("off")
    ax[3].imshow(raster_vis); ax[3].imshow(pred_colored, alpha=0.3); ax[3].set_title("Overlay"); ax[3].axis("off")

    out_path = os.path.join(output_dir, f"vis_{img_name.rsplit('.', 1)[0]}.png")
    print(f"Saving to {out_path}")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

print(f"All visualizations saved to {output_dir}")