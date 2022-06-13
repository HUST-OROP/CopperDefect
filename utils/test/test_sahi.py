#%%
root_path="/home/sunchen/Projects/CopperDetetion"
import sys
sys.path.append(root_path)
import os
os.chdir(root_path)
from mmdet.datasets import build_dataset
from mmcv import Config
from copperdet import *
import json

model_path = "/home/sunchen/Projects/CopperDetetion/archive/Work_dir/20220228/depth/anchor_248/1x1/frcn/best_mAP_epoch_16.pth"
model_config_path = "/home/sunchen/Projects/CopperDetetion/configs/depth/1x1/frcn_depth.py"
model_device = 'cuda:0'
model_confidence_threshold = 0.4
slice_height = 800
slice_width = 800
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "/home/sunchen/Projects/CopperDetetion/dataset/test"
data_config = "/home/sunchen/Projects/CopperDetetion/configs/_base_/datasets/zjdata_1x1split_depth.py"
data_cfg = Config.fromfile(data_config)
slice_data = build_dataset(data_cfg.data.test)

a = slice_data.get_raw_anno_info(slice_data.raw_annofile)
a
# from copperdet.sahi.test_predict import get_sliced_prediction

# # create slices from full image
# slice_image_result = slice_image(
#     image=image,
#     slice_height=slice_height,
#     slice_width=slice_width,
#     overlap_height_ratio=overlap_height_ratio,
#     overlap_width_ratio=overlap_width_ratio,
# )

# num_slices = len(slice_image_result)

# # create prediction input
# num_group = int(num_slices)

# object_prediction_list = []
# # perform sliced prediction
# for group_ind in range(num_group):
    
#     # prepare batch (currently supports only 1 batch)
#     image_list = []
#     shift_amount_list = []
    
#     for image_ind in range(num_batch):
#         image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
#         shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])

#     # perform batch prediction
    
#     # read image as pil
#     image_as_pil = read_image_as_pil(image_list[0])


#%%
image_dir = "/home/sunchen/Projects/CopperDetetion/dataset/test_slice/raw_img"
dataset_json_path = "/home/sunchen/Projects/CopperDetetion/dataset/test_slice/test.json"
output_dir = "/home/sunchen/Projects/CopperDetetion/dataset/test_slice/slice"
min_area_ratio=0.1
ignore_negative_samples=False

def save_json(json_path,json_dict):
    with open(json_path,"w") as fp:
        json.dump(json_dict,fp,indent=4,separators=(",",": "))

from copperdet.sahi.slicing import slice_coco

# assure slice_size is list
slice_size_list = 512
if isinstance(slice_size_list, (int, float)):
    slice_size_list = [slice_size_list]
overlap_ratio = 0.2

# slice coco dataset images and annotations
print("Slicing step is starting...")
for slice_size in slice_size_list:
    # in format: train_images_512_01

    output_images_dir = output_dir
    sliced_coco_name = "DeepPCB_slice"
    
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=dataset_json_path,
        image_dir=image_dir,
        output_coco_annotation_file_name="",
        output_dir=output_images_dir,
        ignore_negative_samples=ignore_negative_samples,
        slice_height=slice_size,
        slice_width=slice_size,
        min_area_ratio=min_area_ratio,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        out_ext=".jpg",
        verbose=False,
    )
    output_coco_annotation_file_path = os.path.join(output_dir, sliced_coco_name + ".json")
    save_json(output_coco_annotation_file_path,coco_dict)
    print(f"Sliced dataset for 'slice_size: {slice_size}' is exported to {output_dir}")
#%%

# %%
