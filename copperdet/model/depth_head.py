import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead

class DepthBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(DepthBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)