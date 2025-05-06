from mmcv import Config
from mmseg.apis import init_segmentor
import torch
cfg = Config.fromfile("configs/sen1floods11_config_prompt_tuning_16.py")
model = init_segmentor(cfg, "/project/hnguyen2/mvu9/folder_04_ma/hls-foundation-os/nprompt_16/best_mIoU_epoch_80.pth", device="cuda")
imgs = torch.randn(1, 6, 1, 224, 224).cuda()
metas = [{
    "img_shape": (224, 224),
    "ori_shape": (224, 224),
    "pad_shape": (224, 224),
    "img_norm_cfg": {"mean": [123.675, 116.28, 103.53, 123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375, 58.395, 57.12, 57.375], "to_rgb": False},
    "scale_factor": 1.0,
    "flip": False,
    "flip_direction": None
}]
try:
    result = model(return_loss=False, img=[imgs], img_metas=metas)
    print("Inference successful:", result)
except Exception as e:
    import traceback
    print(f"Error: {e}\nTraceback:\n{traceback.format_exc()}")