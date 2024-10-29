import os
import time
import shutil
import json
from config import get_config
from easydict import EasyDict as edict
from libs.loss import *
from datasets.ThreeDMatch import ThreeDMatchTrainVal
from datasets.dataloader import get_dataloader
from libs.trainer import Trainer
from torch import optim


# 训练时不固定 seed

if __name__ == '__main__':
    config = get_config()
    dconfig = vars(config)
    for k in dconfig:
        print(f"    {k}: {dconfig[k]}")
    config = edict(dconfig)

    resume = False
    if resume:
        start_epoch = 31
        best_reg_recall = 95.31
        best_F1 = 0.73
        experiment_id = "PointDSC_3DMatch_08121534"
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
    from models.mymodel_better_node import MethodName
    config.mode = "train"
    config.model = MethodName(config)
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
                        in_dim=config.in_dim,
                        inlier_threshold=config.inlier_threshold,
                        num_node=config.num_node, 
                        use_mutual=config.use_mutual,
                        downsample=config.downsample,
                        augment_axis=config.augment_axis,
                        augment_rotation=config.augment_rotation,
                        augment_translation=config.augment_translation,
                        )
    val_set = ThreeDMatchTrainVal(root=config.root,
                            split='val',
                            descriptor=config.descriptor,
                            in_dim=config.in_dim,
                            inlier_threshold=config.inlier_threshold,
                            num_node=config.num_node,
                            use_mutual=config.use_mutual,
                            downsample=config.downsample,
                            augment_axis=config.augment_axis,
                            augment_rotation=config.augment_rotation,
                            augment_translation=config.augment_translation,
                            )
    config.train_loader = get_dataloader(dataset=train_set, 
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        )
    config.val_loader = get_dataloader(dataset=val_set,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        )
    
    # create evaluation
    config.evaluate_metric = {
        "ClassificationLoss": ClassificationLoss(balanced=config.balanced),
        "SpectralMatchingLoss": SpectralMatchingLoss(balanced=config.balanced),
        #"SpectralMatchingLoss": FeatureDifferenceLoss(balanced=config.balanced),
        "TransformationLoss": TransformationLoss(re_thresh=config.re_thre, te_thresh=config.te_thre),
        "HypergraphLoss": EdgeLoss(), #0612
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
        #print("structure: " + str(list(param.size())))
        for j in param.size():
            l *= j
        #print("total: " + str(l))
        k = k + l
    print(k)

    trainer = Trainer(config)
    trainer.train(resume, start_epoch, best_reg_recall, best_F1)
