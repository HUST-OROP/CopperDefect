# %%
### print complete config file
import argparse
import warnings

from mmcv import Config, DictAction

cfg_path = "/home/user/sun_chen/Projects/FSOD/FsMMdet/Model/Model_Fewshot/attention_rpn/neu_det/attention-rpn_r50_c4_4xb2_coco_base-training.py"
fosd_cfg = "/home/user/sun_chen/Projects/FSOD/FsMMdet/Main/mmfewshot/configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.py"
attention_rpn_voc_cfg = "/home/user/sun_chen/Projects/FSOD/FsMMdet/Model/Model_Fewshot/attention_rpn/voc/split1/attention-rpn_r50_c4_voc-split1_base-training.py"
attention_rpn_neu_cfg = "/home/user/sun_chen/Projects/FSOD/FsMMdet/Model/Model_Fewshot/attention_rpn/neu_det/attention-rpn_r50_c4_4xb2_coco_base-training.py"
test_cfg = '/home/user/sun_chen/Projects/FSOD/FsMMdet/Utils/test/./test_config/attention_rpn/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.py'
mspr_cfg = "/home/user/sun_chen/Projects/FsDefect/Model/Model_Fewshot/mpsr/neu_det/mpsr_base_training.py"

tfa_kd_cfg = "/home/user/sun_chen/Projects/FsDefect/Model/Model_Distill/distillers/TFA_distiller/tfa_distiller.py"

def print_cfg(cfg_path):

    cfg = Config.fromfile(cfg_path)

    print(f'Config:\n{cfg.pretty_text}')

print_cfg(tfa_kd_cfg)
# %%
### check checkpoint
import torch
import os
os.chdir("/home/user/sun_chen/Projects/ZJDetection/")

random_init_path = "/home/dlsuncheng/Projects/FSOD/FsMMdet/Weights/base_model_random_init_bbox_head.pth"
base_best_path = "/home/dlsuncheng/Work_dir/FsMMdet/20211201/base_train_10000iter/best_bbox_mAP.pth"
frcn_depth = "/home/user/sun_chen/Projects/ZJDetection/Work_dir/20220107/depth/2x2/frcn/best_mAP.pth"
cascade_depth = "/home/user/sun_chen/Projects/ZJDetection/Work_dir/20220107/depth/3x3/cascade/epoch_24.pth"

remove_ckpt = "/home/user/sun_chen/Projects/FsDefect/Weights/model_reset_remove.pth"

best_tfa = "/home/user/sun_chen/Projects/FsDefect/Weights/base_best_bbox_mAP.pth"

gfl_ckpt = "/home/user/sun_chen/Projects/FsDefect/Weights/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth"

frcn_hrnetw32_ckpt = "./Work_dir/20220228/test/anchor_248/1x1/frcn_hrnet_w32/best_mAP_epoch_1.pth"
def show_weight_param(path):
    print("####################################")
    torch_pth = torch.load(path)
    for param_name in torch_pth["state_dict"].keys():
        if "backbone" in param_name:
            print(param_name)
            # print(torch_pth["state_dict"][param_name].shape)
    print("####################################")

# print("FRCN:")
# show_weight_param(frcn_depth)
# print("Cascade:")
# show_weight_param(cascade_depth)

# show_weight_param(gfl_ckpt)

show_weight_param(frcn_hrnetw32_ckpt)


hrnet_32_ckpt = "/home/user/.cache/torch/hub/checkpoints/hrnetv2_w32-dc9eeb4f.pth"
hrnet_32 = torch.load(hrnet_32_ckpt)

# show_weight_param(base_best_path)
# show_weight_param(random_init_path)
#%%
path = "/home/user/sun_chen/Projects/FSOD/FsMMdet/Utils/test"

atten_rpn_base = "/test_config/attention_rpn/attention-rpn_r50_c4_4xb2_coco_base-training.py"
atten_rpn_10shot = "/test_config/attention_rpn/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.py"
fsce_base = '/test_config/fsce/fsce_r101_fpn_coco_base-training.py'
fsce_10shot = "/test_config/fsce/fsce_r101_fpn_coco_base-training.py"
fsdet_base = "/test_config/fsdetview/fsdetview_r50_c4_8xb4_coco_base-training.py"
fsdet_10shot = "/test_config/fsdetview/fsdetview_r50_c4_8xb4_coco_10shot-fine-tuning.py"
mrcnn_base = "/test_config/meta_rcnn/meta-rcnn_r50_c4_8xb4_coco_base-training.py"
mrcnn_10shot = "/test_config/meta_rcnn/meta-rcnn_r50_c4_8xb4_coco_10shot-fine-tuning.py"
mpsr_base = "/test_config/mpsr/mpsr_r101_fpn_2xb2_coco_base-training.py"
mpsr_10shot = "/test_config/mpsr/mpsr_r101_fpn_2xb2_coco_10shot-fine-tuning.py"

base_config = [atten_rpn_base,fsce_base,fsdet_base,mrcnn_base,mpsr_base]
fs_config = [atten_rpn_10shot,fsce_10shot,fsdet_10shot,mrcnn_10shot,mpsr_10shot]
base_config_list = [path+base_path for base_path in base_config]
fs_config_list = [path+fs_path for fs_path in fs_config]
print(fs_config_list)

# %%
