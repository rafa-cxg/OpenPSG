checkpoint_config = dict(interval=1 ,max_keep_ckpts=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 0), ('val', 1)]
checkpoint_config = dict(interval=300, max_keep_ckpts=10)
evaluation = dict(
    interval=300,
    metric='sgdet',
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
    detection_method='pan_seg',
    save_best='sgdet_score',
    greater_keys='sgdet_score'
)