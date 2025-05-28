# dataset settings
dataset_type = 'VEDAIDataset'
data_root = "/opt/data/private/fcf/dataset/VEDAI/" 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadTwoStreamImageFromFile', spectrals=('rgb', 'ir')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle_m'),   #这个是用于img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadTwoStreamImageFromFile', spectrals=('rgb', 'ir')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle_m'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,  #batchsize
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/rgb/labels/',    
        img_prefix=data_root + 'train/rgb/images/',           
        pipeline=train_pipeline),          
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/rgb/labels/',     
        img_prefix=data_root + 'test/rgb/images/',              
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/rgb/labels/',     
        img_prefix=data_root + 'test/rgb/images/',
        pipeline=test_pipeline))
