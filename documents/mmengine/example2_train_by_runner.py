from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam
from mmengine.runner import Runner
from example1_register import MyAwesomeModel, MyDataset, Accuracy
# from mmengine.registry import MODELS


runner = Runner(
    # your model
    model=MyAwesomeModel(
        layers=2,
        activation='relu'),
    # work directory for saving checkpoints and logs
    work_dir='exp/my_awesome_model',

    # training data
    train_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=True,
            size=10000),
        shuffle=True,
        collate_fn=default_collate,
        batch_size=64,
        pin_memory=True,
        num_workers=2),
    # training configurations
    train_cfg=dict(
        by_epoch=True,   # display in epoch number instead of iterations
        max_epochs=10,
        val_begin=2,     # start validation from the 2nd epoch
        val_interval=1), # do validation every 1 epoch

    # OptimizerWrapper, new concept in MMEngine for richer optimization options
    # Default value works fine for most cases. You may check our documentations
    # for more details, e.g. 'AmpOptimWrapper' for enabling mixed precision
    # training.
    optim_wrapper=dict(
        optimizer=dict(
            type=Adam,
            lr=0.001)),
    # ParamScheduler to adjust learning rates or momentums during training
    param_scheduler=dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1),

    # validation data
    val_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=False,
            size=1000),
        shuffle=False,
        collate_fn=default_collate,
        batch_size=1000,
        pin_memory=True,
        num_workers=2),
    # validation configurations, usually leave it an empty dict
    val_cfg=dict(),
    # evaluation metrics and evaluator
    val_evaluator=dict(type=Accuracy),

    # following are advanced configurations, try to default when not in need
    # hooks are advanced usage, try to default when not in need
    default_hooks=dict(
        # the most commonly used hook for modifying checkpoint saving interval
        checkpoint=dict(type='CheckpointHook', interval=1)),

    # `luancher` and `env_cfg` responsible for distributed environment
    launcher='none',
    env_cfg=dict(
        cudnn_benchmark=False,   # whether enable cudnn_benchmark
        backend='nccl',   # distributed communication backend
        mp_cfg=dict(mp_start_method='fork')),  # multiprocessing configs
    log_level='INFO',

    # load model weights from given path. None for no loading.
    load_from=None,
    # resume training from the given path
    resume=False
)

# start training your model
runner.train()