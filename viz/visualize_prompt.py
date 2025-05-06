import os
import glob
import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv import Config
from mmseg.datasets.pipelines import Compose
from mmcv.utils import Registry
import warnings
warnings.filterwarnings("ignore")

# Registry for custom pipeline transforms
PIPELINES = Registry("pipeline")

# Custom pipeline transform to load images with rasterio
@PIPELINES.register_module()
class LoadImageWithRasterio:
    def __init__(self, to_float32=False, nodata=None, nodata_replace=0):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace

    def __call__(self, results):
        filename = results["img_info"]["filename"]
        with rasterio.open(filename) as src:
            img = src.read()  # Shape: (C, H, W)
        if self.nodata is not None:
            img = np.where(img == self.nodata, self.nodata_replace, img)
        if self.to_float32:
            img = img.astype(np.float32)
        results["img"] = img
        results["img_shape"] = img.shape[1:]  # (H, W)
        results["ori_shape"] = img.shape[1:]  # (H, W)
        return results

# Paths from the experiment
config_path = "configs/sen1floods11_config_prompt_tuning_nprompt_16.py"
work_dir = "/project/hnguyen2/mvu9/folder_04_ma/hls-foundation-os/nprompt_16"
output_dir = os.path.join(work_dir, "inference")
os.makedirs(output_dir, exist_ok=True)

# Find the best mIoU checkpoint
checkpoint_pattern = os.path.join(work_dir, "best_mIoU_epoch_*.pth")
checkpoint_files = glob.glob(checkpoint_pattern)
if not checkpoint_files:
    raise FileNotFoundError("No best_mIoU checkpoint found in {}".format(work_dir))
checkpoint_path = max(checkpoint_files, key=lambda x: int(x.split("_epoch_")[-1].split(".")[0]))
print(f"Using checkpoint: {checkpoint_path}")

# Load the config
cfg = Config.fromfile(config_path)

# Initialize the model
model = init_segmentor(cfg, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")

# Test dataset settings from config
data_root = cfg.data_root
img_dir = cfg.img_dir
ann_dir = cfg.ann_dir
img_suffix = cfg.img_suffix
seg_map_suffix = cfg.seg_map_suffix
test_split = cfg.splits["test"]
ignore_index = cfg.ignore_index
bands = cfg.bands
rgb_bands = [bands.index(i) for i in [1, 2, 3]]  # Indices for RGB bands (1, 2, 3)

# Custom test pipeline using rasterio
test_pipeline = [
    dict(
        type="LoadImageWithRasterio",
        to_float32=False,
        nodata=cfg.image_nodata,
        nodata_replace=cfg.image_nodata_replace,
    ),
    dict(type="BandsExtract", bands=cfg.bands),
    dict(type="ConstantMultiply", constant=cfg.constant),
    dict(type="ToTensor", keys=["img"]),
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **cfg.img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(cfg.bands), cfg.num_frames, -1, -1),
        look_up={"2": 1, "3": 2},
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]
test_pipeline = Compose(test_pipeline)

# Load test split
with open(test_split, "r") as f:
    test_files = [line.strip() for line in f]

# Colormap for visualization
# 0: Background (blue), 1: Flood (red), 2: Ignore/No-data (gray)
palette = {0: [0, 0, 255], 1: [255, 0, 0], 2: [128, 128, 128]}
def apply_colormap(mask, palette, ignore_index):
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in palette:
        colored_mask[mask == cls] = palette[cls]
    return colored_mask

# Visualization function
def visualize_result(img_path, gt_path, pred_mask, output_path):
    # Read input image with rasterio
    with rasterio.open(img_path) as src:
        img = src.read()  # Shape: (C, H, W)
    # Select RGB bands and normalize for display
    rgb_img = img[rgb_bands].transpose(1, 2, 0)  # Shape: (H, W, 3)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6) * 255
    rgb_img = rgb_img.astype(np.uint8)

    # Read ground truth with rasterio
    with rasterio.open(gt_path) as src:
        gt = src.read(1)  # Shape: (H, W)

    # Apply colormap
    gt_colored = apply_colormap(gt, palette, ignore_index)
    pred_colored = apply_colormap(pred_mask, palette, ignore_index)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Input Image (RGB)")
    axes[0].axis("off")
    axes[1].imshow(gt_colored)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    axes[2].imshow(pred_colored)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Process test images
for idx, fname in enumerate(test_files):
    img_path = os.path.join(img_dir, fname + img_suffix)
    gt_path = os.path.join(ann_dir, fname + seg_map_suffix)
    
    # Prepare data for inference
    data = {"img_info": {"filename": img_path}}
    try:
        data = test_pipeline(data)
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        continue
    
    # Run inference
    try:
        result = inference_segmentor(model, [data])
        pred_mask = result[0]  # Shape: (H, W)
    except Exception as e:
        print(f"Error inferring {fname}: {e}")
        continue

    # Save visualization
    output_path = os.path.join(output_dir, f"{fname}_visualization.png")
    print(f"Saving visualization for {fname} to {output_path}")
    visualize_result(img_path, gt_path, pred_mask, output_path)

print(f"Visualizations saved to {output_dir}")