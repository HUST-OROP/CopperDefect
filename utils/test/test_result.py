# %%
from environment import set_envir
set_envir()
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from copperdet import *

path = "/home/sunchen/Projects/CopperDetetion/results.pkl"
# path = "/home/sunchen/Projects/CopperDetetion/trans_result.pkl"
config_path = "/home/sunchen/Projects/CopperDetetion/work_dir/20220618/frcn_hrnet_w40/frcn_hrnet_w40_depth.py"
##%%
results = mmcv.load(path)
# print(results)
cfg = Config.fromfile(config_path)
cfg.data.test.use_slice = True
cfg.data.test.filter_empty_gt=False

data = build_dataset(cfg.data.test)
json_data = data._det2json(results)
## %%
import numpy as np
import torch
from mmdet.core.post_processing import multiclass_nms

raw_datainfo = data.get_raw_anno_info()
raw_name_list = [ann["seg_map"][:-4] for ann in raw_datainfo]
remap_list = {k:[] for k in raw_name_list}
for anno_data in json_data:
    img_name = anno_data["image_name"]
    raw_name,x_1,y_1,x_2,y_2 = img_name.split("_")
    x_1 = float(x_1)
    y_1 = float(y_1)
    
    nms_box = [anno_data["bbox"][0]+x_1,anno_data["bbox"][1]+y_1,
               anno_data["bbox"][2]+anno_data["bbox"][0]+x_1,anno_data["bbox"][3]+anno_data["bbox"][1]+y_1,
               anno_data["score"],1-anno_data["score"],anno_data["depth"]]
    remap_list[raw_name].append(nms_box)
nms_list = []
for k, v in remap_list.items():
    v = np.array(v,dtype=np.float32)
    if len(v.shape)==2:
        bboxes = torch.from_numpy(v[:,:4])
        scores = torch.from_numpy(v[:,4:6])
        heights = torch.from_numpy(v[:,-1]).reshape(-1,1)
        selected_boxes,selected_labels,selected_indx = multiclass_nms(bboxes,scores,score_thr=0.05,
                           nms_cfg=dict(type='nms', iou_threshold=0.5),
                           return_inds=True)
        selected_heights = heights[selected_indx]
        nms_result = torch.concat((selected_boxes,selected_heights),dim=1).numpy()
    else:
        nms_result = np.zeros((0, 6), dtype=np.float32)
    nms_list.append([nms_result])
mmcv.dump(nms_list,"nms_result.pkl")    
# anno_info = data.get_raw_anno_info()
# id2name = data.raw_id2name
# trans_results = []
# for id, name in id2name.items():
    # trans_result = np.array(remap_list[name[:-4]],dtype=np.float32)
#     # print(trans_result)
#     if len(trans_result.shape)==2:
#         trans_result = np.delete(trans_result,[5],axis=1)
#         # print(trans_result)
#     else:
#         trans_result = np.zeros((0, 6), dtype=np.float32)
#     trans_results.append([trans_result])
# # print(trans_results)

# mmcv.dump(nms_list,"trans_result.pkl")
# %%
result_list = ["results.pkl","nms_result.pkl"]
b_result  = mmcv.load(result_list[0])

t_result  = mmcv.load(result_list[1])
#%%
cfg = Config.fromfile(config_path)
cfg.data.test.use_slice = False
raw_data = build_dataset(cfg.data.test)
json_data = raw_data._det2json(t_result)
mmcv.dump(json_data,"trans_result.json")
# import json
# def save_json(json_path,json_dict):
#     with open(json_path,"w") as fp:
#         json.dump(json_dict,fp,indent=4,separators=(",",": "))
# save_json("trans_result.json",json_data[0])
# %%
