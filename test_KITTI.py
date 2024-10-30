import json
import os
import sys
import argparse
import logging
import torch
import numpy as np
import importlib
import open3d as o3d
from tqdm import tqdm
from easydict import EasyDict as edict
from libs.loss import *
from datasets.KITTI import *
from datasets.dataloader import *
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from utils.timer import Timer

set_seed()

def eval_KITTI_per_pair(model, dloader, config, args):
    """
    Evaluate our model on KITTI testset.
    """
    num_pair = dloader.dataset.__len__()
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 12])
    dloader_iter = dloader.__iter__()
    class_loss = ClassificationLoss()
    hyperedge_loss = EdgeFeatureLoss(transpose=True)
    evaluate_metric = TransformationLoss(re_thresh=config.re_thre, te_thresh=config.te_thre)
    data_timer, model_timer = Timer(), Timer()
    H_score, raw_H_score = [], []
    seed_precision = 0
    seed_num = 0
    eval_results = {}
    eval_cnt ={}
    eval_all = 0
    with torch.no_grad():
        for i in tqdm(range(num_pair), ncols=50):
            #################################
            # load data
            #################################
            data_timer.tic()
            corr, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, name = next(dloader_iter)
            seq =int(name[5])
            corr, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels = \
                corr.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), src_normal.cuda(), tgt_normal.cuda(), gt_trans.cuda(), gt_labels.cuda()
            data = {
                'corr_pos': corr,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
                'src_normal': src_normal,
                'tgt_normal': tgt_normal,
                'labels': gt_labels,
                'testing': True,
            }
            data_time = data_timer.toc()

            #################################
            # forward pass
            #################################
            model_timer.tic()
            res = model(data)
            pred_trans, pred_labels = res['final_trans'], res['final_labels']
            confidence, seed = res['confidence'], res['seeds']
            H, edge_score = res['hypergraph'], res['edge_score']
            raw_H = res['raw_H']
            score = hyperedge_loss(H, raw_H, gt_labels).detach().cpu().numpy()
            score1 = hyperedge_loss(raw_H, raw_H, gt_labels).detach().cpu().numpy()
            H_score.append(score)
            raw_H_score.append(score1)

            if args.solver == 'SVD':
                pass

            elif args.solver == 'RANSAC':
                # our method can be used with RANSAC as a outlier pre-filtering step.
                src_pcd = make_point_cloud(src_keypts[0].detach().cpu().numpy())
                tgt_pcd = make_point_cloud(tgt_keypts[0].detach().cpu().numpy())
                corr = np.array([np.arange(src_keypts.shape[1]), np.arange(src_keypts.shape[1])])
                pred_inliers = np.where(pred_labels.detach().cpu().numpy() > 0)[1]
                corr = o3d.utility.Vector2iVector(corr[:, pred_inliers].T)
                reg_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    src_pcd, tgt_pcd, corr,
                    max_correspondence_distance=config.inlier_threshold,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=5000, max_validation=5000)
                )
                inliers = np.array(reg_result.correspondence_set)
                pred_labels = torch.zeros_like(gt_labels)
                pred_labels[0, inliers[:, 0]] = 1
                pred_trans = torch.eye(4)[None].to(src_keypts.device)
                pred_trans[:, :4, :4] = torch.from_numpy(reg_result.transformation)

            if args.use_icp:
                pred_trans = icp_refine(src_keypts, tgt_keypts, src_normal, tgt_normal, pred_trans, config.inlier_threshold)

            model_time = model_timer.toc()
            class_stats = class_loss(pred_labels, gt_labels)
            recall, Re, Te, rmse = evaluate_metric(pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)

            seed_precision += gt_labels[:, seed].sum() / seed.size(1)
            seed_num += gt_labels[:, seed].sum()

            # save statistics
            stats[i, 0] = float(recall / 100.0)  # success
            stats[i, 1] = float(Re)  # Re (deg)
            stats[i, 2] = float(Te)  # Te (cm)
            stats[i, 3] = int(torch.sum(gt_labels))  # input inlier number
            stats[i, 4] = float(torch.mean(gt_labels.float()))  # input inlier ratio
            stats[i, 5] = int(torch.sum(gt_labels[pred_labels > 0]))  # output inlier number
            stats[i, 6] = float(class_stats['precision'])  # output inlier precision
            stats[i, 7] = float(class_stats['recall'])  # output inlier recall
            stats[i, 8] = float(class_stats['f1'])  # output inlier f1 score
            stats[i, 9] = model_time
            stats[i, 10] = data_time
            stats[i, 11] = -1

            if recall == 0:
                from evaluation.benchmark_utils import rot_to_euler
                R_gt, t_gt = gt_trans[0][:3, :3], gt_trans[0][:3, -1]
                euler = rot_to_euler(R_gt.detach().cpu().numpy())

                input_ir = float(torch.mean(gt_labels.float()))
                input_i = int(torch.sum(gt_labels))
                output_i = int(torch.sum(gt_labels[pred_labels > 0]))
                logging.info(
                    f"Pair {i}, GT Rot: {euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}, Trans: {t_gt[0]:.2f}, {t_gt[1]:.2f}, {t_gt[2]:.2f}, RE: {float(Re):.2f}, TE: {float(Te):.2f}")
                logging.info((
                                 f"\tInput Inlier Ratio :{input_ir * 100:.2f}%(#={input_i}), Output: IP={float(class_stats['precision']) * 100:.2f}%(#={output_i}) IR={float(class_stats['recall']) * 100:.2f}%"))

            if not seq in eval_results:
                eval_results[seq] = 0
                eval_cnt[seq] = 0
            eval_results[seq] += recall / 100.0
            eval_cnt[seq] += 1
            eval_all += recall / 100.0
            torch.cuda.empty_cache()
            with open(f'logs/{args.dataset}/' + config.descriptor + '.txt', 'a') as f:
                lines = [str(i), ' ',
                         str(stats[i][0]), ' ', str(stats[i][1]), ' ', str(stats[i][2]),
                         '\n', str(pred_trans[0].detach().cpu().numpy()), '\n']
                f.writelines(lines)

    print(seed_precision / num_pair)
    print(seed_num / num_pair)

    logging.info(f"All sequences: {eval_all / num_pair * 100:.2f}")
    logging.info("Each sequence:")
    for seq in eval_results:
        logging.info(f"Sequence {seq}: {eval_results[seq] / eval_cnt[seq] * 100:.2f}")

    import matplotlib.pyplot as plt
    H_data = np.array(H_score)
    raw_H_data = np.array(raw_H_score)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(raw_H_data, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], color='skyblue', edgecolor='black',
             alpha=0.7)
    ax1.set_title('raw')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency')
    ax1.set_ylim(0, 500)
    ax2.hist(H_data, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], color='skyblue', edgecolor='black',
             alpha=0.7)
    ax2.set_title('processed')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency')
    ax2.set_ylim(0, 500)
    plt.tight_layout()
    plt.savefig(f'snapshot/{args.chosen_snapshot}/KITTI_{config.descriptor}.png')
    #plt.show()
    return stats


def eval_KITTI(model, config, args):
    dset = KITTIDatasetTest(root=config.root,
                        split='test',
                        descriptor=config.descriptor,
                        inlier_threshold=config.inlier_threshold,
                        num_node=8000,
                        use_mutual=config.use_mutual,
                        augment_axis=0,
                        augment_rotation=0.00,
                        augment_translation=0.0,
                        args=args
                        )
    dloader = get_dataloader_lc(dset, batch_size=1, num_workers=8, shuffle=False)
    os.makedirs(f'logs/{args.dataset}', exist_ok=True)
    if os.path.isfile(f'logs/{args.dataset}/{config.descriptor}.txt'):
        os.remove(f'logs/{args.dataset}/{config.descriptor}.txt')
        print("File Deleted successfully")
    stats = eval_KITTI_per_pair(model, dloader, config, args)
    logging.info(f"Max memory allicated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

    # pair level average
    allpair_stats = stats
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(
        f"All {allpair_stats.shape[0]} pairs, Mean Success Rate={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4] * 100:.2f}%)")
    logging.info(
        f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6] * 100:.2f}%, recall={allpair_average[7] * 100:.2f}%, f1={allpair_average[8] * 100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    return allpair_stats


if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='HyperGCT_KITTI_release', type=str, help='snapshot dir')
    parser.add_argument('--dataset', default='KITTI_10m', type=str, choices=['KITTI_10m', 'KITTI_LC'])
    parser.add_argument('--range', default='0_10', type=str, choices=['0_10', '10_20', '20_30'])
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--use_icp', default=False, type=str2bool)
    args = parser.parse_args()
    print(args.chosen_snapshot, args.dataset)
    if args.use_icp:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}-ICP.log'
    else:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}.log'

    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    os.makedirs(f'logs/{args.dataset}', exist_ok=True)

    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    config.re_thre = 5
    config.te_thre = 60
    if args.dataset == 'KITTI_LC':
        config.inlier_threshold = 1.2
        config.seed_ratio = 1.0
        if args.range == "10_20":
            config.te_thre = 120
        elif args.range == "20_30":
            config.te_thre = 180
    elif args.dataset == 'KITTI_10m':
        config.inlier_threshold = 0.6
    # change the dataset path here
    # config.root = ''
    config.descriptor = 'fpfh'
    config.mode = 'test'
    from models.mymodel import MethodName
    model = MethodName(config)

    miss = model.load_state_dict(torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best.pkl'), strict=False)
    print(miss)
    model.eval()

    # evaluate on the test set
    stats = eval_KITTI(model.cuda(), config, args)
