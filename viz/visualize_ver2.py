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
import traceback
from tqdm import tqdm
import signal
from contextlib import contextmanager

warnings.filterwarnings("ignore")

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

@PIPELINES.register_module()
class LoadImageWithRasterio:
    def __init__(self, to_float32=False, nodata=None, nodata_replace=0, resize=(224, 224), io_timeout=30):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace
        self.resize = resize
        self.io_timeout = io_timeout

    def __call__(self, results):
        filename = results["img_info"]["filename"]
        try:
            with timeout(self.io_timeout):
                with rasterio.open(filename) as src:
                    img = src.read()
                    band_count = src.count
        except Exception as e:
            raise Exception(f"Failed to open {filename}: {e}")

        if band_count < 13:
            raise ValueError(f"Image {filename} has {band_count} bands, expected at least 13")

        if self.nodata is not None:
            img = np.where(img == self.nodata, self.nodata_replace, img)
        if self.to_float32:
            img = img.astype(np.float32)

        img_resized = np.stack([cv2.resize(band, self.resize, interpolation=cv2.INTER_LINEAR) for band in img])

        results.update({
            "img": img_resized,
            "img_shape": img_resized.shape[1:],
            "ori_shape": img_resized.shape[1:],
            "pad_shape": img_resized.shape[1:],
            "scale_factor": 1.0,
            "flip": False,
            "flip_direction": None,
            "filename": filename,
            "ori_filename": os.path.basename(filename),
            "img_info": {"filename": filename},
        })
        return results

@PIPELINES.register_module()
class CustomBandsExtract:
    def __init__(self, bands):
        self.bands = bands

    def __call__(self, results):
        img = results["img"]
        img = img[self.bands]
        results["img"] = img
        return results

@PIPELINES.register_module()
class CustomConstantMultiply:
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, results):
        img = results["img"]
        constant = np.array(self.constant, dtype=img.dtype)
        if constant.shape == (img.shape[0],):
            constant = constant[:, np.newaxis, np.newaxis]
        results["img"] = img * constant
        return results

@PIPELINES.register_module()
class CustomNormalize:
    def __init__(self, mean, std, to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = results["img"]
        mean = self.mean[:, np.newaxis, np.newaxis]
        std = self.std[:, np.newaxis, np.newaxis]
        results["img"] = (img - mean) / std
        results["img_norm_cfg"] = {"mean": self.mean.tolist(), "std": self.std.tolist(), "to_rgb": self.to_rgb}
        return results

def custom_inference_segmentor(model, data):
    model.eval()
    with torch.no_grad():
        imgs = data["img"]
        metas = data["img_metas"]
        if isinstance(metas, DataContainer):
            metas = metas.data[0] if isinstance(metas.data, list) else metas.data
        if isinstance(imgs, DataContainer):
            imgs = imgs.data
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        imgs = [img.unsqueeze(0).cuda() if torch.cuda.is_available() else img.unsqueeze(0) for img in imgs]
        return model(return_loss=False, img=imgs, img_metas=[[metas]])

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
    return np.moveaxis(np.stack(channels), 0, -1)[..., :3][..., ::-1]

def apply_colormap(mask, palette):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in palette.items():
        color[mask == cls] = rgb
    return color

# Load config and model
config_path = "configs/sen1floods11_config_prompt_tuning_16.py"
checkpoint_path = "/project/hnguyen2/mvu9/folder_04_ma/hls-foundation-os/nprompt_16/best_mIoU_epoch_80.pth"
cfg = Config.fromfile(config_path)
cfg.img_norm_cfg = {
    'mean': [123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    'to_rgb': False
}
cfg.constant = 0.0001
model = init_segmentor(cfg, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu")

bands = cfg.bands
resize_shape = (224, 224)
rgb_bands = [bands.index(i) for i in [1, 2, 3]]

# Pipeline
pipeline_cfg = [
    dict(type="LoadImageWithRasterio", to_float32=False, nodata=cfg.image_nodata, nodata_replace=cfg.image_nodata_replace, resize=resize_shape),
    dict(type="CustomBandsExtract", bands=cfg.bands),
    dict(type="CustomConstantMultiply", constant=cfg.constant),
    dict(type="CustomNormalize", **cfg.img_norm_cfg),
    dict(type="ToTensor", keys=["img"]),
    dict(type="Reshape", keys=["img"], new_shape=(len(cfg.bands), cfg.num_frames, -1, -1), look_up={"2": 1, "3": 2}),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(type="Collect", keys=["img"], meta_keys=[
        "img_info", "filename", "ori_filename", "img_shape", "ori_shape", "img_norm_cfg",
        "pad_shape", "scale_factor", "flip", "flip_direction"]),
]
test_pipeline = Compose(pipeline_cfg)

# Visualization Setup
palette = {0: [0, 0, 255], 1: [255, 0, 0], 2: [128, 128, 128]}
output_dir = "./vis/outputs/nprompt_16_resized"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("/project/hnguyen2/mvu9/datasets/SEN1Floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv")
img_list, mask_list = df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()
img_dir, ann_dir = cfg.img_dir, cfg.ann_dir

skip_set = set([
    "Somalia_60129_S1Hand.tif", "USA_504150_S1Hand.tif", "USA_758178_S1Hand.tif",
    "USA_670826_S1Hand.tif", "USA_595451_S1Hand.tif", "Paraguay_80102_S1Hand.tif"
])

filtered = [(i, m) for i, m in zip(img_list, mask_list) if i not in skip_set]
img_list, mask_list = zip(*filtered)

# Inference loop
for idx, (img_name, mask_name) in tqdm(list(enumerate(zip(img_list, mask_list))), total=len(img_list)):
    img_name_s2 = img_name.replace("S1", "S2")
    img_path = os.path.join(img_dir, img_name_s2)
    label_path = os.path.join(ann_dir, mask_name)

    try:
        data = {"img_info": {"filename": img_path}}
        for t in test_pipeline.transforms:
            data = t(data)
    except Exception as e:
        print(f"[ERROR] Failed preprocessing {img_name_s2}: {e}")
        continue

    try:
        result = custom_inference_segmentor(model, data)
        pred_mask = result[0]
    except Exception as e:
        print(f"[ERROR] Inference failed for {img_name_s2}: {e}")
        continue

    try:
        with rasterio.open(img_path) as src:
            raw = src.read()
        raw = np.where(raw == cfg.image_nodata, cfg.image_nodata_replace, raw)
        raw = np.stack([cv2.resize(b, resize_shape) for b in raw])
        rgb = enhance_raster_for_visualization(raw[rgb_bands])
    except Exception as e:
        print(f"[ERROR] Failed to load or preprocess image: {e}")
        continue

    try:
        with rasterio.open(label_path) as src:
            gt = src.read(1)
        gt = cv2.resize(gt, resize_shape, interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"[ERROR] Failed to load label: {e}")
        continue

    fig, ax = plt.subplots(1, 4, figsize=(15, 10))
    ax[0].imshow(rgb); ax[0].set_title("Input RGB")
    ax[1].imshow(apply_colormap(gt, palette)); ax[1].set_title("Ground Truth")
    ax[2].imshow(apply_colormap(pred_mask, palette)); ax[2].set_title("Prediction")
    ax[3].imshow(rgb); ax[3].imshow(apply_colormap(pred_mask, palette), alpha=0.3); ax[3].set_title("Overlay")
    for a in ax: a.axis('off')
    plt.savefig(os.path.join(output_dir, f"vis_{os.path.splitext(img_name_s2)[0]}.png"), bbox_inches="tight", dpi=300)
    plt.close()

print(f"\n✅ All visualizations saved to: {output_dir}")
