# dataset settings
dataset_type = 'PanopticSceneGraphDataset'
# ann_file = './data/psg/psg.json' # full data, available after PSG challenge
ann_file = './data/psg/psg_train_val.json'  # './data/psg/psg_train_val.json' for PSG challenge development
# ann_file = './data/psg/psg_val_test.json' # for PSG challenge submission
coco_root = './data/coco'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
dataset_resample= dict(type='BGNN_Resample',method='bilvl',RESAMPLING_PARAM=dict(REPEAT_FACTOR=0.012,INSTANCE_DROP_RATE=1.2))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticSceneGraphAnnotations',
        with_bbox=True,
        with_rel=True,
        with_mask=True,
        with_seg=True,
    ),

    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='SceneGraphFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img',
            'gt_bboxes',
            'gt_labels',
            'gt_rels',
            'gt_relmaps',
            'gt_masks',
            'gt_semantic_seg',
        ],
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # Since the forward process may need gt info, annos must be loaded.
    dict(type='LoadPanopticSceneGraphAnnotations',
         with_bbox=True,
         with_rel=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # NOTE: Do not change the img to DC.
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            dict(
                type='ToDataContainer',
                fields=(dict(key='gt_bboxes'), dict(key='gt_labels')),
            ),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ],
    ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=coco_root,
        seg_prefix=coco_root,
        pipeline=train_pipeline,
        split='train',
        all_bboxes=True,
        resample= dataset_resample
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=coco_root,
        seg_prefix=coco_root,
        pipeline=test_pipeline,
        split='test',
        all_bboxes=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=coco_root,
        seg_prefix=coco_root,
        pipeline=test_pipeline,
        split='test',
        all_bboxes=True,
    ),
)
