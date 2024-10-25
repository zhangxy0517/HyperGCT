import json
import os.path
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
from datasets.ThreeDMatch_bac import ThreeDMatchTest
from datasets.dataloader_bac import get_dataloader
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from evaluation.benchmark_utils_predator import *
from utils.timer import Timer
set_seed()

H_score = []
raw_H_score = []

def eval_3DMatch_scene(model, scene, scene_ind, dloader, config, args):
    """
    Evaluate our model on 3DMatch testset [scene]
    """
    correct_num = 0
    correct_ratio = 0
    seed_precision = 0
    seed_num = 0
    num_pair = dloader.dataset.__len__()
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number 
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 13])
    dloader_iter = dloader.__iter__()
    class_loss = ClassificationLoss()
    hyperedge_loss = EdgeFeatureLoss(transpose=True)
    evaluate_metric = TransformationLoss(re_thresh=config.re_thre, te_thresh=config.te_thre)
    data_timer, model_timer = Timer(), Timer()


    final_poses = np.zeros([num_pair, 4, 4])
    with torch.no_grad():
        for i in tqdm(range(num_pair), ncols=100):
            #################################
            # load data 
            #################################
            data_timer.tic()
            corr, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, scene, src_id, tgt_id = next(dloader_iter)
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
            # corr, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, relax_idx, relax_match_points, relax_dis, src_pts = next(
            #     dloader_iter)
            # corr, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, relax_idx, relax_match_points, relax_dis, src_pts = \
            #     corr.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), src_normal.cuda(), tgt_normal.cuda(), gt_trans.cuda(), gt_labels.cuda(), relax_idx.cuda(), relax_match_points.cuda(), relax_dis.cuda(), src_pts.cuda()
            # data = {
            #     'corr_pos': corr,
            #     'src_keypts': src_keypts,
            #     'tgt_keypts': tgt_keypts,
            #     'src_normal': src_normal,
            #     'tgt_normal': tgt_normal,
            #     'labels': gt_labels,
            #     'relax_idx': relax_idx,
            #     'relax_match_points': relax_match_points,
            #     'relax_dis': relax_dis,
            #     'src_pts': src_pts,
            #     'testing': True,
            # }
            data_time = data_timer.toc()

            #################################
            # forward pass 
            #################################
            model_timer.tic()
            res = model(data)
            model_time = model_timer.toc()
            pred_trans, pred_labels = res['final_trans'], res['final_labels']
            H, confidence = res['hypergraph'], res['confidence']
            raw_H = res['raw_H']
            edge_score, seed = res['edge_score'], res['seeds']


            sampled_trans = res['sampled_trans']
            sampled_re, sampled_te = transformation_error(sampled_trans, gt_trans)
            succ = torch.where(sampled_re < config.re_thre) and torch.where(sampled_te < config.te_thre)
            recall = len(succ[0])
            correct_num += recall
            correct_ratio += recall / sampled_trans.shape[0] * 100.0

            bs, num, _ = raw_H.size()

            # topk, _ = torch.topk(edge_score, k=round(num * 0.1), dim=-1)  # 每个节点选出topk概率的超边，该超边包含该节点
            # a_min = torch.min(topk, dim=-1).values.unsqueeze(-1).repeat(1, 1, num)
            # W = torch.where(torch.greater_equal(edge_score, a_min), edge_score, torch.zeros_like(edge_score))
            # H = torch.where(torch.greater(W, 0), H, torch.zeros_like(H))

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
                #corr = o3d.utility.Vector2iVector(corr.T)
                reg_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    src_pcd, tgt_pcd, corr,
                    max_correspondence_distance=config.inlier_threshold,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=4000000)
                )
                inliers = np.array(reg_result.correspondence_set)
                pred_labels = torch.zeros_like(gt_labels)
                pred_labels[0, inliers[:, 0]] = 1
                pred_trans = torch.eye(4)[None].to(src_keypts.device)
                pred_trans[:, :4, :4] = torch.from_numpy(reg_result.transformation)

            if args.use_icp:
                pred_trans = icp_refine(src_keypts, tgt_keypts, src_normal, tgt_normal, pred_trans, config.inlier_threshold)


            class_stats = class_loss(pred_labels, gt_labels)
            recall, Re, Te, rmse = evaluate_metric(pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)

            seed_precision += gt_labels[:, seed].sum().item() / seed.size(1) * 100.0
            seed_num += gt_labels[:, seed].sum().item()
            
            #################################
            # record the evaluation results.
            #################################
            # save statistics
            stats[i, 0] = float(recall / 100.0)                      # success
            stats[i, 1] = float(Re)                                  # Re (deg)
            stats[i, 2] = float(Te)                                  # Te (cm)
            stats[i, 3] = int(torch.sum(gt_labels))                  # input inlier number
            stats[i, 4] = float(torch.mean(gt_labels.float()))       # input inlier ratio
            stats[i, 5] = int(torch.sum(gt_labels[pred_labels > 0])) # output inlier number TODO (predict) 不是越高越好
            stats[i, 6] = float(class_stats['precision'])            # output inlier precision
            stats[i, 7] = float(class_stats['recall'])               # output inlier recall
            stats[i, 8] = float(class_stats['f1'])                   # output inlier f1 score
            stats[i, 9] = model_time
            stats[i, 10] = data_time
            stats[i, 11] = int(src_id)
            stats[i, 12] = int(tgt_id)
            final_poses[i] = pred_trans[0].detach().cpu().numpy()
            torch.cuda.empty_cache()

    return final_poses, stats, seed_num, seed_precision, correct_num, correct_ratio

def eval_3DMatch(model, config, args):
    """
    Collect the evaluation results on each scene of 3DMatch testset, write the result to a .log file.
    """
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    all_stats = {}
    all_poses = None
    avg_seed_num = 0
    avg_seed_precision = 0
    avg_correct_num = 0
    avg_correct_ratio = 0
    if os.path.isfile('logs/3dmatch/'+args.descriptor+'.txt'):
        os.remove('logs/3dmatch/'+args.descriptor+'.txt')
        print("File Deleted successfully")
    for scene_ind, scene in enumerate(scene_list):
        dset = ThreeDMatchTest(root='/data/zxy/Threedmatch_dataset',
                               descriptor=args.descriptor,
                               in_dim=config.in_dim,
                               inlier_threshold=config.inlier_threshold,
                               num_node=args.num_points,
                               use_mutual=config.use_mutual,
                               augment_axis=0,
                               augment_rotation=0.00,
                               augment_translation=0.0,
                               select_scene=scene,
                               )
        dloader = get_dataloader(dset, batch_size=1, num_workers=8, shuffle=False)
        scene_poses, scene_stats, seed_num, seed_precision, correct_num, correct_ratio = eval_3DMatch_scene(model, scene, scene_ind, dloader, config, args)
        avg_seed_num += seed_num
        avg_seed_precision += seed_precision
        avg_correct_num += correct_num
        avg_correct_ratio += correct_ratio
        if scene_ind == 0:
            all_poses = scene_poses
        else:
            all_poses = np.concatenate([all_poses, scene_poses], axis=0)
        all_stats[scene] = scene_stats
        # save to file
        with open('logs/3dmatch/'+args.descriptor+'.txt','a') as f:
            for i in range(len(scene_poses)):
                lines = [scene,' ' ,'cloud_bin_'+str(int(scene_stats[i][11]))+'+cloud_bin_'+str(int(scene_stats[i][12])), ' ',str(scene_stats[i][0]),' ', str(scene_stats[i][1]),' ', str(scene_stats[i][2]),'\n', str(scene_poses[i]),'\n']
                f.writelines(lines)

    logging.info(f"Max memory allicated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

    # result for each scene
    scene_vals = np.zeros([len(scene_list), 13])
    scene_ind = 0
    for scene, stats in all_stats.items():
        correct_pair = np.where(stats[:, 0] == 1)
        scene_vals[scene_ind] = stats.mean(0)
        # for Re and Te, we only average over the successfully matched pairs.
        scene_vals[scene_ind, 1] = stats[correct_pair].mean(0)[1]
        scene_vals[scene_ind, 2] = stats[correct_pair].mean(0)[2]
        logging.info(f"Scene {scene_ind}th:"
                     f" Reg Recall={scene_vals[scene_ind, 0] * 100:.2f}% "
                     f" Mean RE={scene_vals[scene_ind, 1]:.2f} "
                     f" Mean TE={scene_vals[scene_ind, 2]:.2f} "
                     f" Mean Precision={scene_vals[scene_ind, 6] * 100:.2f}% "
                     f" Mean Recall={scene_vals[scene_ind, 7] * 100:.2f}% "
                     f" Mean F1={scene_vals[scene_ind, 8] * 100:.2f}%"
                     )
        scene_ind += 1

    # scene level average
    average = scene_vals.mean(0)
    logging.info(f"All {len(scene_list)} scenes, Mean Reg Recall={average[0] * 100:.2f}%, Mean Re={average[1]:.2f}, Mean Te={average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={average[3]:.2f}(ratio={average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={average[5]:.2f}(precision={average[6] * 100:.2f}%, recall={average[7] * 100:.2f}%, f1={average[8] * 100:.2f}%)")
    logging.info(f"\tMean model time: {average[9]:.2f}s, Mean data time: {average[10]:.2f}s")

    # pair level average 
    stats_list = [stats for _, stats in all_stats.items()]
    allpair_stats = np.concatenate(stats_list, axis=0)
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Reg Recall={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6] * 100:.2f}%, recall={allpair_average[7] * 100:.2f}%, f1={allpair_average[8] * 100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    # inlier ratio level average
    # if args.descriptor == "fpfh":
    #     inlier_thresh = 0.05
    # elif args.descriptor == "fcgf":
    #     inlier_thresh = 0.1
    #
    # below_thresh_pairs = allpair_stats[allpair_stats[:, 4] <= inlier_thresh]
    # below_thresh_pairs_average = below_thresh_pairs.mean(0)
    # correct_below_thresh_pairs_average = below_thresh_pairs[below_thresh_pairs[:, 0] == 1].mean(0)
    # logging.info(
    #     f"All {below_thresh_pairs.shape[0]} pairs, Mean Reg Recall={below_thresh_pairs_average[0] * 100:.2f}%, Mean Re={correct_below_thresh_pairs_average[1]:.2f}, Mean Te={correct_below_thresh_pairs_average[2]:.2f}")

    font1 = {
        'family':'Times New Roman',
        'weight':'normal',
        'size':28,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 24,
    }

    import matplotlib.pyplot as plt
    # plt.rc('font', family='Times New Roman')
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.letax.preample'] = r'\usepackage{}'
    H_data = np.array(H_score)
    raw_H_data = np.array(raw_H_score)
    width = 0.04
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.hist(raw_H_data + 0.05 - width, bins=bins + 0.05 - width, color='skyblue', edgecolor='black',
             alpha=0.7, label='initial graph', width=width, align='mid')
    ax.hist(H_data + 0.05, bins=bins + 0.05, color='orange', edgecolor='black',
             alpha=0.7, label='output graph', width=width, align='mid')
    #ax.set_title('Raw vs Processed Score Distribution')
    ax.set_xlabel('Score',font1)
    ax.set_ylabel('Frequency', font1)
    ax.set_xlim(-0.02, 1)
    ax.set_ylim(0, 500)
    ax.legend(prop=font2)
    plt.xticks(bins, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.tick_params(labelsize=24)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.tight_layout()
    plt.savefig(f'snapshot/{args.chosen_snapshot}/Match_{args.descriptor}_{args.num_points}.png')
    plt.show()

    all_stats_npy = np.concatenate([v for k, v in all_stats.items()], axis=0)
    print(avg_seed_num / 1623, avg_seed_precision / 1623)
    print(avg_correct_num / 1623, avg_correct_ratio / 1623)
    return all_stats_npy


if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='HyperGCT_3DMatch_release', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--descriptor', default='fcgf', type=str, choices=['fcgf', 'fpfh'])
    parser.add_argument('--num_points', default='all', type=str)
    parser.add_argument('--use_icp', default=False, type=str2bool)
    args = parser.parse_args()
    print(args.chosen_snapshot)
    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    config.inlier_threshold = 0.1
    config.sigma_d = 0.1
    config.re_thre = 15
    config.te_thre = 30


    if args.use_icp:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}-{config.descriptor}-ICP.log'
    else:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}-{config.descriptor}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    config.mode = "test"
    from snapshot.HyperGCT_3DMatch_release.mymodel import MethodName
    model = MethodName(config)
    miss = model.load_state_dict(torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best.pkl'), strict=True)
    print(miss)

    model.eval()

    params = list(model.parameters())
    k = 0
    for param in params:
        l = 1
        #print("structure: " + str(list(param.size())))
        for j in param.size():
            l *= j
        #print("total: " + str(l))
        k = k + l
    print(k)

    # evaluate on the test set
    stats = eval_3DMatch(model.cuda(), config, args)

    # claculate area under the cumulative error curve.
    # re_auc = exact_auc(stats[:, 1], thresholds=[5, 10, 15])
    # te_auc = exact_auc(stats[:, 2], thresholds=[5, 10, 15, 20, 25, 30])
    # print(f"RE AUC:", re_auc)
    # print(f"TE AUC:", te_auc)
