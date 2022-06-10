import mmcv
import numpy as np
import pytest
import torch

from mmdet.core import bbox2roi
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmdet.models.roi_heads.depth_roi_head import DepthRoIHead
from mmdet.models.roi_heads.bbox_heads import DepthBBoxHead
from mmdet.core.bbox.assigners import MaxIoUAssignerDepth
from mmdet.core.bbox.samplers import RandomSamplerDepth
import torch

from mmdet.core import build_assigner, build_sampler

def _dummy_bbox_sampling(proposal_list, gt_bboxes, gt_labels,gt_depth = None):
    """Create sample results that can be passed to BBoxHead.get_targets."""
    num_imgs = 1
    feat = torch.rand(1, 1, 3, 3)
    assign_config = dict(
        type='MaxIoUAssignerDepth',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        ignore_iof_thr=-1)
    sampler_config = dict(
        type='RandomSamplerDepth',
        num=512,
        pos_fraction=0.25,
        neg_pos_ub=-1,
        add_gt_as_proposals=True)
    bbox_assigner = build_assigner(assign_config)
    bbox_sampler = build_sampler(sampler_config)
    gt_bboxes_ignore = [None for _ in range(num_imgs)]
    sampling_results = []
    for i in range(num_imgs):
        assign_result = bbox_assigner.assign(proposal_list[i], gt_bboxes[i],
                                             gt_bboxes_ignore[i], gt_labels[i],gt_depth[i])
        sampling_result = bbox_sampler.sample(
            assign_result,
            proposal_list[i],
            gt_bboxes[i],
            gt_labels[i],
            gt_depth[i],
            feats=feat)
        sampling_results.append(sampling_result)

    return sampling_results

#%%
"""Tests bbox head loss when truth is empty and non-empty."""
self = DepthBBoxHead(in_channels=8, roi_feat_size=3,num_classes=1)

# Dummy proposals
proposal_list = [
    torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
]

target_cfg = mmcv.Config(dict(pos_weight=1))

# Test bbox loss when truth is empty
gt_bboxes = [
    torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
]
gt_labels = [torch.LongTensor([0])]

gt_depth = [torch.LongTensor([48])]

sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                        gt_labels,gt_depth)

bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,gt_depth,
                                target_cfg)

labels, label_weights, bbox_targets, bbox_weights,depth = bbox_targets

# Create dummy features "extracted" for each sampled bbox
num_sampled = sum(len(res.bboxes) for res in sampling_results)
rois = bbox2roi([res.bboxes for res in sampling_results])
dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)

cls_scores, bbox_preds,depth_preds = self.forward(dummy_feats)

losses = self.loss(cls_scores, bbox_preds,depth_preds,rois, labels, label_weights,
                    bbox_targets, bbox_weights,depth)
#%%

self = DepthBBoxHead(in_channels=8, roi_feat_size=3,num_classes=1,reg_class_agnostic=True)
num_sample = 1 #0
num_class = 6
rois = torch.rand((num_sample, 5))
cls_score = torch.rand((num_sample, num_class))
bbox_pred = torch.rand((num_sample, 4))

#?
depth_pred = torch.rand((num_sample,2))


scale_factor = np.array([2.0, 2.0, 2.0, 2.0])
det_bboxes, det_labels = self.get_bboxes(
    rois, cls_score, bbox_pred, None, scale_factor, rescale=True)
if num_sample == 0:
    assert len(det_bboxes) == 0 and len(det_labels) == 0
else:
    assert det_bboxes.shape == bbox_pred.shape
    assert det_labels.shape == cls_score.shape