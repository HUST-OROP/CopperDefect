#%%
from environment import set_envir
set_envir()
from copperdet import *

import pickle
from mmcv import Config
from mmdet.datasets import build_dataset
import time 
from mean_ap import eval_map_depth
from collections import OrderedDict
from mmcv.utils import print_log
import mmcv

result_path = "/home/sunchen/Projects/CopperDetetion/results.pkl"
results = mmcv.load(result_path)
config_path = "/home/sunchen/Projects/CopperDetetion/configs/depth/frcn_hrnet_w40_depth.py"
cfg = Config.fromfile(config_path)

cfg.data.test.use_slice = True
cfg.data.test.filter_empty_gt=False

# cfg.data.test.use_slice = False

dataset = build_dataset(cfg.data.test)
post_dict=dict(postproscess_type='soft_nms', iou_threshold=0.5, min_score=0.05)
dataset.evaluate(results,standard_eval=True,postprocess=True,postprocess_cfg=post_dict)

#%%
#### Bbox & Depth result
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
config_file = "/home/user/sun_chen/Projects/ZJDetection/Model/full_data/frcn_depth.py"
cfg = Config.fromfile(config_file)
##GT result
datasets = build_dataset(cfg.data.test)
annotations = [datasets.get_ann_info(i) for i in range(len(datasets))]

## pred result
result_file_path = "/home/user/sun_chen/Projects/ZJDetection/Result/cascade_depth.pkl"
f = open(result_file_path,'rb')
results = pickle.load(f)
bbox_result = [[img_res[0][:,:-1]] for img_res in results]
depth_result = [img_res[0][:,-1] for img_res in results]
depth_gt = [anno["depth"] for anno in annotations]

#%%
################
mean_aps = []
eval_results = OrderedDict()
iou_thrs = [0.5]

for iou_thr in iou_thrs:
    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
    mean_ap, _ = eval_map_depth(
        results,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=("lizi",),
        logger=None,
        use_legacy_coordinate=True)
    mean_aps.append(mean_ap)
    eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
eval_results['mAP'] = sum(mean_aps) / len(mean_aps)    
