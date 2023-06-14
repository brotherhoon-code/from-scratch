# Save the following codes in example_config.py
# Almost copied from the above example, with some commas removed
model = dict(type='MyAwesomeModel',
    layers=2,
    activation='relu')
work_dir = 'exp/my_awesome_model'

train_dataloader = dict(
    dataset=dict(type='MyDataset',
        is_train=True,
        size=10000),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    pin_memory=True,
    num_workers=2)
train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_begin=2,
    val_interval=1)
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.001))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[4, 8],
    gamma=0.1)

val_dataloader = dict(
    dataset=dict(type='MyDataset',
        is_train=False,
        size=1000),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=1000,
    pin_memory=True,
    num_workers=2)
val_cfg = dict()
val_evaluator = dict(type='Accuracy')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))
launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    backend='nccl',
    mp_cfg=dict(mp_start_method='fork'))
log_level = 'INFO'
load_from = None
resume = False