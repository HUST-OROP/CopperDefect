#%%
import torch

from mmdet.core.bbox.assigners import MaxIoUAssigner,MaxIoUAssignerDepth
from mmdet.core.bbox.iou_calculators import BboxOverlaps2D,build_iou_calculator

self = MaxIoUAssigner(
    pos_iou_thr=0.5,
    neg_iou_thr=0.5,
    match_low_quality=False
)

self_depth = MaxIoUAssignerDepth(
    pos_iou_thr=0.5,
    neg_iou_thr=0.5,
    match_low_quality=False
)

bboxes = torch.FloatTensor([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
    [5, 5, 15, 15],
    [32, 32, 38, 42],
])
gt_bboxes = torch.FloatTensor([
    [0, 0, 10, 9],
    [0, 10, 10, 19],
])
gt_labels = torch.LongTensor([2, 3])
gt_depth  = torch.LongTensor([101, 102])

# iou_cal_config = dict(type='BboxOverlaps2D')
# iou_calculator = build_iou_calculator(iou_cal_config)
# iou_result = iou_calculator(gt_bboxes, bboxes)
# print(iou_result)
# assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)

assign_result = self_depth.assign(bboxes, gt_bboxes, gt_labels=gt_labels,gt_depth=gt_depth)

# assert len(assign_result.gt_inds) == 4
# assert len(assign_result.labels) == 4

# expected_gt_inds = torch.LongTensor([1, 0, 2, 0])
# assert torch.all(assign_result.gt_inds == expected_gt_inds)
# %%
import torch
x = torch.arange(0,20)*5
y = torch.arange(0,20)*5
xx = x.repeat(len(y))
yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
aa = torch.stack([xx,yy,xx,yy],dim=-1)
print(aa)
# %%
