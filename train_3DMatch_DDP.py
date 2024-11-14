import os
import shutil
import json
from config import get_config
from easydict import EasyDict as edict
from libs.loss import *
from datasets.ThreeDMatch import ThreeDMatchTrainVal
from datasets.dataloader import get_dataloader_train
from libs.trainer import Trainer
from torch import optim
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

# 训练时不固定 seed

def main_worker(rank, world_size, config, resume, start_epoch, best_reg_recall, best_F1):
    # Set the device for each process
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    from models.mymodel import MethodName
    config.mode = "train"
    config.model = MethodName(config).to(rank)
    config.model = nn.parallel.DistributedDataParallel(config.model, device_ids=[rank], find_unused_parameters=True)

    # create optimizer
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            # momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    # create dataset and dataloader
    train_set = ThreeDMatchTrainVal(root=config.root,
                                    descriptor=config.descriptor,
                                    split='train',
                                    inlier_threshold=config.inlier_threshold,
                                    num_node=config.num_node,
                                    downsample=config.downsample,
                                    augment_axis=config.augment_axis,
                                    augment_rotation=config.augment_rotation,
                                    augment_translation=config.augment_translation,
                                    )
    val_set = ThreeDMatchTrainVal(root=config.root,
                                  split='val',
                                  descriptor=config.descriptor,
                                  inlier_threshold=config.inlier_threshold,
                                  num_node=config.num_node,
                                  downsample=config.downsample,
                                  augment_axis=config.augment_axis,
                                  augment_rotation=config.augment_rotation,
                                  augment_translation=config.augment_translation,
                                  )

    # DistributedSampler ensures each process has its subset of data
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    config.train_loader = get_dataloader_train(dataset=train_set,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               sampler=train_sampler
                                               )
    config.val_loader = get_dataloader_train(dataset=val_set,
                                             batch_size=config.batch_size,
                                             num_workers=config.num_workers,
                                             )

    # create evaluation
    config.evaluate_metric = {
        "ClassificationLoss": ClassificationLoss(balanced=config.balanced).to(rank),
        "SpectralMatchingLoss": SpectralMatchingLoss(balanced=config.balanced).to(rank),
        "TransformationLoss": TransformationLoss(re_thresh=config.re_thre, te_thresh=config.te_thre).to(rank),
        "HypergraphLoss": EdgeLoss().to(rank),  # 0612
    }
    config.metric_weight = {
        "ClassificationLoss": config.weight_classification,
        "SpectralMatchingLoss": config.weight_spectralmatching,
        "TransformationLoss": config.weight_transformation,
        "HypergraphLoss": config.weight_hypergraph,
    }

    params = list(config.model.parameters())
    k = 0
    for param in params:
        l = 1
        # print("structure: " + str(list(param.size())))
        for j in param.size():
            l *= j
        # print("total: " + str(l))
        k = k + l
    print(k)

    trainer = Trainer(config)
    trainer.train(resume, start_epoch, best_reg_recall, best_F1)

    # Clean up distributed process group
    dist.destroy_process_group()

if __name__ == '__main__':
    config = get_config()
    dconfig = vars(config)
    for k in dconfig:
        print(f"    {k}: {dconfig[k]}")
    config = edict(dconfig)
    # change the dataset path here
    #config.root = '/media/SSD/PCR_methods/Threedmatch_dataset'

    resume = False
    if resume:
        start_epoch = 0
        best_reg_recall = 0
        best_F1 = 0
        experiment_id = ""
        config.snapshot_dir = f'snapshot/{experiment_id}'
        config.tboard_dir = f'tensorboard/{experiment_id}'
        config.save_dir = os.path.join(f'snapshot/{experiment_id}', 'models/')
    else:
        start_epoch = 0
        best_reg_recall = 0
        best_F1 = 0

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    shutil.copy2(os.path.join('.', 'train_3DMatch.py'), os.path.join(config.snapshot_dir, 'train.py'))
    shutil.copy2(os.path.join('.', 'libs/trainer.py'), os.path.join(config.snapshot_dir, 'trainer.py'))
    shutil.copy2(os.path.join('.', 'models/mymodel.py'), os.path.join(config.snapshot_dir, 'mymodel.py'))  # for the model setting.
    shutil.copy2(os.path.join('.', 'libs/loss.py'), os.path.join(config.snapshot_dir, 'loss.py'))
    shutil.copy2(os.path.join('.', 'datasets/ThreeDMatch.py'), os.path.join(config.snapshot_dir, 'dataset.py'))
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )

    # Distributed setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    world_size = torch.cuda.device_count()
    config.world_size = world_size
    # Run main_worker on each available GPU
    mp.spawn(main_worker, args=(world_size, config, resume, start_epoch, best_reg_recall, best_F1), nprocs=world_size)
