import argparse
import time
import os

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


dataset = '3DMatch'
experiment_id = f"HyperGCT_{dataset}_{time.strftime('%m%d%H%M')}"
# snapshot configurations
snapshot_arg = add_argument_group('Snapshot')
snapshot_arg.add_argument('--snapshot_dir', type=str, default=f'snapshot/{experiment_id}')
snapshot_arg.add_argument('--tboard_dir', type=str, default=f'tensorboard/{experiment_id}')
snapshot_arg.add_argument('--snapshot_interval', type=int, default=1)
snapshot_arg.add_argument('--save_dir', type=str, default=os.path.join(f'snapshot/{experiment_id}', 'models/'))

# Loss configurations
loss_arg = add_argument_group('Loss')
loss_arg.add_argument('--evaluate_interval', type=int, default=1)
loss_arg.add_argument('--balanced', type=str2bool, default=False)
loss_arg.add_argument('--weight_classification', type=float, default=1.0)
loss_arg.add_argument('--weight_spectralmatching', type=float, default=1.0)
loss_arg.add_argument('--weight_hypergraph', type=float, default=1.0)
loss_arg.add_argument('--weight_transformation', type=float, default=0.0)
loss_arg.add_argument('--transformation_loss_start_epoch', type=int, default=0)

# Optimizer configurations
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'])
opt_arg.add_argument('--max_epoch', type=int, default=50)
opt_arg.add_argument('--training_max_iter', type=int, default=3500)
opt_arg.add_argument('--val_max_iter', type=int, default=1000)
opt_arg.add_argument('--lr', type=float, default=1e-4)
opt_arg.add_argument('--weight_decay', type=float, default=1e-6)
opt_arg.add_argument('--momentum', type=float, default=0.9)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--scheduler_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler_interval', type=int, default=1)

# Dataset and dataloader configurations
data_arg = add_argument_group('Data')
if dataset == '3DMatch':
    data_arg.add_argument('--root', type=str, default='/data/zxy/Threedmatch_dataset')
    data_arg.add_argument('--descriptor', type=str, default='fcgf', choices=['fpfh', 'fcgf'])
    data_arg.add_argument('--inlier_threshold', type=float, default=0.10)
    data_arg.add_argument('--downsample', type=float, default=0.03)
    data_arg.add_argument('--re_thre', type=float, default=15, help='rotation error thrshold (deg)')
    data_arg.add_argument('--te_thre', type=float, default=30, help='translation error thrshold (cm)')
else:
    data_arg.add_argument('--root', type=str, default='/data/zxy/KITTI')
    data_arg.add_argument('--descriptor', type=str, default='fcgf', choices=['fcgf', 'fpfh'])
    data_arg.add_argument('--inlier_threshold', type=float, default=1.2)
    data_arg.add_argument('--downsample', type=float, default=0.30)
    data_arg.add_argument('--re_thre', type=float, default=5, help='rotation error thrshold (deg)')
    data_arg.add_argument('--te_thre', type=float, default=60, help='translation error thrshold (cm)')

data_arg.add_argument('--seed_ratio', type=float, default=0.2, help='max ratio of seeding points')
data_arg.add_argument('--num_node', type=int, default=1000)
data_arg.add_argument('--augment_axis', type=int, default=3)
data_arg.add_argument('--augment_rotation', type=float, default=1.0, help='rotation angle = num * 2pi')
data_arg.add_argument('--augment_translation', type=float, default=0.5, help='translation = num (m)')
data_arg.add_argument('--batch_size', type=int, default=6)
data_arg.add_argument('--num_workers', type=int, default=12)

# Other configurations
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--gpu_mode', type=str2bool, default=True)
misc_arg.add_argument('--verbose', type=str2bool, default=True)
misc_arg.add_argument('--pretrain', type=str, default='')
misc_arg.add_argument('--weights_fixed', type=str2bool, default=False)


def get_config():
    args = parser.parse_args()
    return args
