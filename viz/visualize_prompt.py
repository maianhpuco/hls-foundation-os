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
from mmcv.parallel import DataContainer
import cv2
import warnings

warnings.filterwarnings("ignore")

# --- Custom Raster Loader ---
@PIPELINES.register_module()
class LoadImageWithRasterio:
    def __init__(self, to_float32=False, nodata=None, nodata_replace=0, resize=(224, 224)):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace
        self.resize = resize

    def __call__(self, results):
        filename = results["img_info"]["filename"]
        try:
            with rasterio.open(filename) as src:
                img = src.read()  # (C, H, W)
                band_count = src.count
        except Exception as e:
            raise Exception(f"Failed to open {filename}: {e}")
        
        # Validate band count
        if band_count < 13:
            raise ValueError(f"Image {filename} has {band_count} bands, expected at least 13")

        if self.nodata is not None:
            img = np.where(img == self.nodata, self.nodata_replace, img)
        if self.to_float32:
            img = img.astype(np.float32)

        # Resize to (C, 224, 224)
        img_resized = np.stack([cv2.resize(band, self.resize, interpolation=cv2.INTER_LINEAR)
                                for band in img])

        results["img"] = img_resized
        results["img_shape"] = img_resized.shape[1:]
        results["ori_shape"] = img_resized.shape[1:]
        results["pad_shape"] = img_resized.shape[1:]
        results["scale_factor"] = 1.0
        results["flip"] = False
        results["flip_direction"] = None
        results["filename"] = filename
        results["ori_filename"] = os.path.basename(filename)
        results["img_info"] = {"filename": filename}
        return results

# --- Custom BandsExtract ---
@PIPELINES.register_module()
class CustomBandsExtract:
    def __init__(self, bands):
        self.bands = bands

    def __call__(self, results):
        img = results["img"]
        print(f"Bands to extract: {self.bands}")
        if img.shape[0] < max(self.bands) + 1:
            raise ValueError(f"Image has {img.shape[0]} bands, expected at least {max(self.bands) + 1}")
        img = img[self.bands]
        if img.shape[0] != len(self.bands):
            raise ValueError(f"Expected {len(self.bands)} bands, got {img.shape[0]}")
        results["img"] = img
        return results

# --- Custom ConstantMultiply ---
@PIPELINES.register_module()
class CustomConstantMultiply:
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, results):
        img = results["img"]
        print(f"CustomConstantMultiply: constant={self.constant}, constant_shape={np.shape(self.constant) if isinstance(self.constant, np.ndarray) else 'scalar'}, img_shape={img.shape}")
        
        # Convert constant to NumPy array if scalar
        if np.isscalar(self.constant):
            constant = np.array(self.constant, dtype=img.dtype)
        else:
            constant = np.array(self.constant, dtype=img.dtype)

        # Ensure constant is broadcastable
        if constant.shape == ():
            pass  # Scalar, no reshaping needed
        elif constant.shape == (img.shape[0],):
            constant = constant[:, np.newaxis, np.newaxis]  # Shape (C, 1, 1)
        elif constant.shape != img.shape:
            raise ValueError(f"Constant shape {constant.shape} is not broadcastable to img shape {img.shape}")

        # Perform multiplication using NumPy
        try:
            results["img"] = img * constant
        except Exception as e:
            raise ValueError(f"Failed to multiply img with constant: {e}")

        return results

# --- Custom Normalize ---
@PIPELINES.register_module()
class CustomNormalize:
    def __init__(self, mean, std, to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.img_norm_cfg = {'mean': mean, 'std': std, 'to_rgb': to_rgb}

    def __call__(self, results):
        img = results["img"]
        print(f"CustomNormalize: mean_shape={self.mean.shape}, std_shape={self.std.shape}, img_shape={img.shape}, to_rgb={self.to_rgb}")
        
        # Ensure mean and std are broadcastable
        if self.mean.shape != (img.shape[0],) or self.std.shape != (img.shape[0],):
            raise ValueError(f"Mean shape {self.mean.shape} or std shape {self.std.shape} does not match img channels {img.shape[0]}")
        
        # Reshape mean and std for broadcasting
        mean = self.mean[:, np.newaxis, np.newaxis]  # Shape (C, 1, 1)
        std = self.std[:, np.newaxis, np.newaxis]    # Shape (C, 1, 1)

        # Perform normalization using NumPy
        try:
            img = (img - mean) / std
        except Exception as e:
            raise ValueError(f"Failed to normalize img: {e}")

        results["img"] = img
        results["img_norm_cfg"] = self.img_norm_cfg  # Add img_norm_cfg to results
        return results

# --- Inference Helper ---
def custom_inference_segmentor(model, data):
    model.eval()
    try:
        with torch.no_grad():
            imgs = data["img"]
            metas = data["img_metas"]
            print(f"Image metas type: {type(metas)}")
            if isinstance(metas, str):
                raise ValueError(f"img_metas is a string: {metas}")
            if isinstance(imgs, DataContainer):
                imgs = imgs.data
            if isinstance(metas, DataContainer):
                metas = metas.data
            if isinstance(imgs, torch.Tensor):
                imgs = [imgs]
            imgs = [img.unsqueeze(0).cuda() if torch.cuda.is_available() else img.unsqueeze(0) for img in imgs]
            result = model(return_loss=False, img=imgs, img_metas=[metas])
        return result
    except Exception as e:
        raise Exception(f"Inference failed: {str(e)}")

# --- Visualization Helpers ---
def enhance_raster_for_visualization(raster):
    NO_DATA_FLOAT = 0.0001
    PERCENTILES = (0.1, 99.9)
    channels = []
    for channel in range(raster.shape[0]):
        ref = raster[channel]
        valid_mask = ref != NO_DATA_FLOAT
        if not np.any(valid_mask):
            channels.append(np.zeros_like(ref))
            continue
        vmin, vmax = np.percentile(ref[valid_mask], PERCENTILES)
        norm = (ref - vmin) / (vmax - vmin)
        norm[~valid_mask] = 0
        channels.append(np.clip(norm, 0, 1))
    rgb = np.moveaxis(np.stack(channels), 0, -1)[..., :3][..., ::-1]
    return rgb

def apply_colormap(mask, palette):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        colored[mask == cls] = color
    return colored

# --- Config Setup ---
config_path = "configs/sen1floods11_config_prompt_tuning_16.py"
checkpoint_path = "/project/hnguyen2/mvu9/folder_04_ma/hls-foundation-os/nprompt_16/best_mIoU_epoch_80.pth"
output_dir = "./vis/outputs/nprompt_16_resized"
split_csv_path = "/project/hnguyen2/mvu9/datasets/SEN1Floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv"
os.makedirs(output_dir, exist_ok=True)

cfg = Config.fromfile(config_path)
# Debug config parameters
print(f"Original config img_norm_cfg: {Config.fromfile(config_path).img_norm_cfg}")
print(f"Config img_norm_cfg: {cfg.img_norm_cfg}")
print(f"Config constant: {cfg.constant}, constant_shape={np.shape(cfg.constant) if isinstance(cfg.constant, (np.ndarray, list)) else 'scalar'}")
print(f"Config num_frames: {cfg.num_frames}")

# Override img_norm_cfg to ensure 6 channels
cfg.img_norm_cfg = {
    'mean': [123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    'to_rgb': False
}

# Override constant to ensure compatibility
cfg.constant = 0.0001  # Scalar value

model = init_segmentor(cfg, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")

bands = cfg.bands
print(f"Config bands: {bands}")
rgb_bands = [bands.index(i) for i in [1, 2, 3]]
resize_shape = (224, 224)

# --- Define Pipeline ---
test_pipeline = [
    dict(type="LoadImageWithRasterio", to_float32=False, nodata=cfg.image_nodata, nod dupe_replace=cfg.image_nodata_replace, resize=resize_shape),
    dict(type="CustomBandsExtract", bands=cfg.bands),
    dict(type="CustomConstantMultiply", constant=cfg.constant),
    dict(type="CustomNormalize", **cfg.img_norm_cfg),
    dict(type="ToTensor", keys=["img"]),
    dict(type="Reshape", keys=["img"], new_shape=(len(cfg.bands), cfg.num_frames, -1, -1), look_up={"2": 1, "3": 2}),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(type="Collect", keys=["img"], meta_keys=[
        "img_info", "filename", "ori_filename", "img_shape", "ori_shape", "img_norm_cfg",
        "pad_shape", "scale_factor", "flip", "flip_direction"
    ]),
]
test_pipeline = Compose(test_pipeline)

# --- Colormap ---
palette = {0: [0, 0, 255], 1: [255, 0, 0], 2: [128, 128, 128]}

# --- Load Dataset ---
df = pd.read_csv(split_csv_path)
img_list = df.iloc[:, 0].tolist()
mask_list = df.iloc[:, 1].tolist()
img_dir = cfg.img_dir
ann_dir = cfg.ann_dir

# --- Run Inference ---
for idx, img_name in enumerate(img_list):
    print(f"\n--- {idx} - {img_name} ---")
    mask_name = mask_list[idx]
    img_name_s2 = img_name.replace("S1", "S2")
    img_path = os.path.join(img_dir, img_name_s2)
    label_path = os.path.join(ann_dir, mask_name)

    try:
        data = {"img_info": {"filename": img_path}}
        # Debug shape and type after each transform
        for i, transform in enumerate(test_pipeline.transforms):
            data = transform(data)
            if "img" in data:
                shape = data["img"].shape if isinstance(data["img"], (torch.Tensor, np.ndarray)) else "Not a tensor/array"
                dtype = type(data["img"]).__name__
                print(f"Shape after transform {i} ({transform.__class__.__name__}): {shape}, Type: {dtype}")
            # Debug available keys before Collect
            if i == len(test_pipeline.transforms) - 2:  # Before Collect
                print(f"Keys before Collect: {list(data.keys())}")
        print(f"Final pipeline keys: {list(data.keys())}")
        print(f"Final image shape: {data['img'].shape if isinstance(data['img'], torch.Tensor) else 'Not a tensor'}")
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
        input_data = np.stack([cv2.resize(b, resize_shape) for b in input_data])
        rgb_image = enhance_raster_for_visualization(input_data[rgb_bands])
    except Exception as e:
        print(f"Error loading image {img_name_s2}: {e}")
        continue

    try:
        with rasterio.open(label_path) as src:
            gt_mask = src.read(1)
        gt_mask = cv2.resize(gt_mask, resize_shape, interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"Error loading label {mask_name}: {e}")
        continue

    gt_color = apply_colormap(gt_mask, palette)
    pred_color = apply_colormap(pred_mask, palette)

    fig, ax = plt.subplots(1, 4, figsize=(15, 10))
    ax[0].imshow(rgb_image)
    ax[0].set_title("Input RGB")
    ax[1].imshow(gt_color)
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_color)
    ax[2].set_title("Prediction")
    ax[3].imshow(rgb_image)
    ax[3].imshow(pred_color, alpha=0.3)
    ax[3].set_title("Overlay")
    for a in ax:
        a.axis('off')

    fname = os.path.splitext(img_name_s2)[0]
    plt.savefig(os.path.join(output_dir, f"vis_{fname}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

print(f"\nAll visualizations saved to {output_dir}")