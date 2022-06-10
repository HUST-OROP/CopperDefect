#%%
from json import load
import os
import torch
import pickle
ckpt_path = "/home/user/sun_chen/Projects/ZJDetection/Work_dir/20220102/depth/cascade/best_mAP.pth"
ckpt_file = torch.load(ckpt_path)

depth_result_path ="/home/user/sun_chen/Projects/ZJDetection/Result/depth_results.pkl"
depth_result_f = open(depth_result_path,"rb")
depth_result_file = pickle.load(depth_result_f)

result_path ="/home/user/sun_chen/Projects/ZJDetection/Result/bbox_results.pkl"
result_f = open(result_path,"rb")
result_file = pickle.load(result_f)

result_file
# %%
import torchsummaryX as summary
import torch.nn as nn
import torch

from mmdet.apis import init_detector
import mmcv

class ConvNet(nn.Module):
    def __init__(self,config):
        super(ConvNet, self).__init__()
        self.config = mmcv.Config.fromfile(config)
        self.model= init_detector(self.config)

    def forward(self,x):
        result = self.model.forward_dummy(x)
        return result

def load_ckpt(ckpt_path):
    
    ckpt = torch.load(ckpt_path)
    model = ckpt["state_dict"]
    return model

frcn_r101_path = "/home/user/sun_chen/Projects/ZJDetection/Work_dir/20220222/depth/3x3/frcn_r101/epoch_24.pth"
frcn_hrnet_w32_path = "/home/user/sun_chen/Projects/ZJDetection/Work_dir/20220222/depth/3x3/frcn_hrnet_w32/epoch_24.pth"
frcn_hrnet_w40_path = "/home/user/sun_chen/Projects/ZJDetection/Work_dir/20220222/depth/3x3/frcn_hrnet_w40/epoch_24.pth"

frcn_r101 = load_ckpt(frcn_r101_path)
frcn_hrnet_w32 = load_ckpt(frcn_hrnet_w32_path)
frcn_hrnet_w40 = load_ckpt(frcn_hrnet_w40_path)


input = torch.zeros([1 ,3, 200, 200])
summary(frcn_r101,input)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# input = input.to(device)
# input.to(device)
# MyConvNet = ConvNet()
# MyConvNet.model.to(device)
# y = MyConvNet(input)
# %%
