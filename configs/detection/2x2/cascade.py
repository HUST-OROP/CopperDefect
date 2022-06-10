_base_ = [
    "../../_base_/models/cascade_rcnn_r50_fpn.py",
    "../../_base_/datasets/zjdata_2x2split.py",
    "../../_base_/schedules/schedule_1x.py",
    "../../_base_/default_runtime.py"
]

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

evaluation = dict(interval=1, metric='mAP',save_best='mAP')

# model = dict(roi_head=dict(bbox_head=dict(loss_depth = dict(type="L1Loss",loss_weight=0.05))))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[17, 19])
runner = dict(type='EpochBasedRunner', max_epochs=24)

checkpoint_config = dict(interval=100)
