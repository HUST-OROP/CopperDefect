# dataset settings
dataset_type = 'ZJDepthDataset'

data_root = './dataset/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleWithDepth'),
    dict(type='Collect', keys=['img','gt_bboxes','gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "/COCO-Annotations_depth/trainval.json",
        img_prefix=data_root + 'Images/',
        sliced_image_folder = '/home/sunchen/Projects/CopperDetetion/dataset/slice/train_images',
        sliced_anno_path = '/home/sunchen/Projects/CopperDetetion/dataset/slice/slice_train.json',
        slice_process=True,
        use_slice=True,
        filter_empty_gt=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'COCO-Annotations_depth/test.json',
        img_prefix=data_root + 'Images/',
        sliced_image_folder = '/home/sunchen/Projects/CopperDetetion/dataset/slice/test_images',
        sliced_anno_path = '/home/sunchen/Projects/CopperDetetion/dataset/slice/slice_test.json',
        slice_process=False,
        use_slice=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'COCO-Annotations_depth/test.json',
        img_prefix=data_root + 'Images/',
        sliced_image_folder = '/home/sunchen/Projects/CopperDetetion/dataset/slice/test_images',
        sliced_anno_path = '/home/sunchen/Projects/CopperDetetion/dataset/slice/slice_test.json',
        slice_process=False,
        use_slice=True,
        pipeline=test_pipeline))
