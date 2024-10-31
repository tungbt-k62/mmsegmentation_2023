dataset_type = 'BKAI_Dataset'
data_root = '/Users/tungbui/Desktop/AI_labs/bkai_dataset/fold_1'

# Định nghĩa train_pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),  # Load ảnh gốc
    dict(type='LoadAnnotations'),  # Load nhãn
    dict(type='Resize', img_scale=(384, 384), keep_ratio=True),  # Resize ảnh
    dict(type='RandomFlip', flip_ratio=0.5),  # Lật ngẫu nhiên với xác suất 0.5
    dict(type='PhotoMetricDistortion'),  # Biến đổi màu sắc ngẫu nhiên
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),  # Chuẩn hóa ảnh
    dict(type='Pad', size=(384, 384), pad_val=0, seg_pad_val=255),  # Padding ảnh
    dict(type='DefaultFormatBundle'),  # Định dạng bundle chuẩn
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])  # Tập hợp các key cần thiết cho training
]

# Định nghĩa test_pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline)
)
