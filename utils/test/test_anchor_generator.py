#%%
from mmcv import ConfigDict
from mmdet.core.anchor import AnchorGenerator
from mmdet.core.anchor import build_anchor_generator
ag_config=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32,64])
# ag_cfg = ConfigDict(ag_config)
# self = build_anchor_generator(ag_cfg)
self = AnchorGenerator([8],[0.5,1,2.0],[2,4,8,16,32,64])
self.num_base_anchors
# %%
