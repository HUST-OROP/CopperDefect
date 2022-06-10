#%%
ROOT = "/home/user/sun_chen/Projects/reconstruct_zj"
import sys
import os
os.chdir(ROOT)
sys.path.append(ROOT)

from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmcv import Config
from zjdet import *
cfg_path = "/home/user/sun_chen/Projects/reconstruct_zj/configs/depth/1x1/frcn_depth.py"

cfg = Config.fromfile(cfg_path)
model = build_detector(cfg.model)
dataset = build_dataset(cfg.data.train)
# %%
