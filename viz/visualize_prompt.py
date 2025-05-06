import os
import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
import pandas as pd
from mmcv import Config
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose
from mmseg.datasets.builder import PIPELINES
import warnings
warnings.filterwarnings("ignore")

# Custom pipeline transform to load raster images
@PIPELINES.register_module()
class LoadImageWithRasterio:
    def __init__(self, to_float32=False, nodata=None, nodata_replace=0):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace

    def __call__(self, results):
        filename = results["img_info"]["filename"]
        with rasterio.open(filename) as src:
            img = src.read()
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

# Inference helper
def custom_inference_segmentor(model, data):
    model.eval()
    with torch.no_grad():
        imgs = data["img"]
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]  # wrap single tensor in list
        imgs = [img.unsqueeze(0) for img in imgs]  # [B, C, H, W]
        imgs = [img.cuda() if torch.cuda.is_available() else img for img in imgs]
        img_metas = [data["img_metas"]]
        result = model(return_loss=False, img=imgs, img_metas=img_metas)
    return result

# Visualization helper
def enhance_raster_for_visualization(raster, ref_img=None):
    NO_DATA_FLOAT = 0.0001
    PERCENTILES = (0.1, 99.9)
    if ref_img is None:
        ref_img = raster
    channels = []
    for channel in range(raster.shape[0]):
        valid_mask = np.ones_like(ref_img[channel], dtype=bool)
        valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
        mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
        normalized = (raster[channel] - mins) / (maxs - mins)
        normalized[~valid_mask] = 0
        channels.append(np.clip(normalized, 0, 1))
    rgb = np.moveaxis(np.stack(channels), 0, -1)[..., :3][..., ::-1]
    return rgb

# Colormap
palette = {0: [0, 0, 255], 1: [255, 0, 0], 2: [128, 128, 128]}
def apply_colormap(mask, palette, ignore_index):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        colored[mask == cls] = color
    return colored

# --- Configs ---
config_path = "configs/sen1floods11_config_prompt_tuning_16.py"
checkpoint_path = "/project/hnguyen2/mvu9/folder_04_ma/hls-foundation-os/nprompt_16/best_mIoU_epoch_80.pth"
output_dir = "./vis/outputs/nprompt_16"
os.makedirs(output_dir, exist_ok=True)
split_csv_path = "/project/hnguyen2/mvu9/datasets/SEN1Floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv"

cfg = Config.fromfile(config_path)
model = init_segmentor(cfg, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")

# Required fields
img_dir = cfg.img_dir
ann_dir = cfg.ann_dir
bands = cfg.bands
rgb_bands = [bands.index(i) for i in [1, 2, 3]]

# Build test pipeline
test_pipeline = Compose([
    dict(type="LoadImageWithRasterio", to_float32=False, nodata=cfg.image_nodata, nodata_replace=cfg.image_nodata_replace),
    dict(type="BandsExtract", bands=cfg.bands),
    dict(type="ConstantMultiply", constant=cfg.constant),
    dict(type="ToTensor", keys=["img"]),
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **cfg.img_norm_cfg),
    dict(
        type="Reshape", keys=["img"],
        new_shape=(len(cfg.bands), cfg.num_frames, -1, -1),
        look_up={"2": 1, "3": 2},
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info", "filename", "ori_filename", "img_shape", "ori_shape", "img_norm_cfg"
        ],
    )
])

# Load test list
df = pd.read_csv(split_csv_path)
img_list = df.iloc[:, 0].tolist()
mask_list = df.iloc[:, 1].tolist()

for idx, img_name in enumerate(img_list):
    print(f"\n--- {idx} - {img_name} ---")
    mask_name = mask_list[idx]
    img_name_s2 = img_name.replace("S1", "S2")
    img_path = os.path.join(img_dir, img_name_s2)
    label_path = os.path.join(ann_dir, mask_name)

    data = {"img_info": {"filename": img_path}}
    try:
        data = test_pipeline(data)
        print(f"Results keys after pipeline for {img_name_s2}: {list(data.keys())}")
    except Exception as e:
        print(f"Error processing {img_name_s2}: {e}")
        continue

    try:
        result = custom_inference_segmentor(model, data)
        pred_mask = result[0]
    except Exception as e:
        print(f"Error inferring {img_name_s2}: {e}")
        continue

    try:
        with rasterio.open(img_path) as src:
            input_data = src.read()
        input_data = np.where(input_data == cfg.image_nodata, cfg.image_nodata_replace, input_data)
        raster_vis = enhance_raster_for_visualization(input_data[rgb_bands])
    except Exception as e:
        print(f"Error loading input for {img_name_s2}: {e}")
        continue

    try:
        with rasterio.open(label_path) as src:
            label_data = src.read(1)
    except Exception as e:
        print(f"Error loading label for {mask_name}: {e}")
        continue

    gt_colored = apply_colormap(label_data, palette, cfg.ignore_index)
    pred_colored = apply_colormap(pred_mask, palette, cfg.ignore_index)

    fig, ax = plt.subplots(1, 4, figsize=(15, 10))
    ax[0].imshow(raster_vis)
    ax[0].set_title("Input RGB")
    ax[1].imshow(gt_colored)
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_colored)
    ax[2].set_title("Prediction")
    ax[3].imshow(raster_vis)
    ax[3].imshow(pred_colored, alpha=0.3)
    ax[3].set_title("Overlay")

    for a in ax:
        a.axis('off')

    save_name = os.path.splitext(img_name_s2)[0]
    save_path = os.path.join(output_dir, f"vis_{save_name}.png")
    print(f"Saving: {save_path}")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

print(f"\nDone. Visualizations saved to: {output_dir}")
