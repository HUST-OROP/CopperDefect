#%%
import pickle
import numpy as np
from mmcv import Config
from mmcv.utils import config
from mmdet.datasets import build_dataset,build_dataloader
import time 
from mmdet.datasets import ZJDataset
from mean_ap import get_cls_results,eval_map_depth
from collections import OrderedDict
from mmcv.utils import print_log

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
#%%
# depth_gt = [anno["depth"] for anno in annotations]
# depth_pred = [np.zeros(anno["depth"].shape) for anno in annotations]

# for i in range(len(depth_gt)):
#     for gt_index,pred_index in gt2pred[i].items():
#         depth_pred[i][gt_index]=depth_result[i][pred_index]

# gt2pred

# %%
#### Normal Bbox result
# timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
# config_file = "/home/user/sun_chen/Projects/ZJDetection/Model/full_data/hrnet.py"
# cfg = Config.fromfile(config_file)
# ##GT result
# datasets = build_dataset(cfg.data.test)
# annotations = [datasets.get_ann_info(i) for i in range(len(datasets))]
# ## pred result
# result_file_path = "/home/user/sun_chen/Projects/ZJDetection/Result/bbox_results.pkl"
# f = open(result_file_path,'rb')
# results = pickle.load(f)
################
