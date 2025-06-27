import json
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
from datasets.ThreeDMatch import ThreeDLOMatchTest
from datasets.dataloader import get_dataloader
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from evaluation.benchmark_utils_predator import *
from utils.timer import Timer
from utils.SE3 import *
set_seed()


def eval_3DMatch_scene(model, scene_ind, dloader, config, args):
    correct_num = 0
    correct_ratio = 0
    seed_precision = 0
    seed_num = 0
    num_pair = dloader.dataset.__len__()
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number 
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 13])
    final_poses = np.zeros([num_pair, 4, 4])
    dloader_iter = dloader.__iter__()
    class_loss = ClassificationLoss()
    hyperedge_loss = EdgeFeatureLoss(transpose=True)
    evaluate_metric = TransformationLoss(re_thresh=config.re_thre, te_thresh=config.te_thre)
    data_timer, model_timer = Timer(), Timer()
    H_score, raw_H_score = [], []
    with (torch.no_grad()):
        for i in tqdm(range(num_pair), ncols=100):
            #################################
            # load data 
            #################################
            data_timer.tic()

            corr, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, scene, src_id, tgt_id = next(
                dloader_iter)
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
            H, edge_score = res['hypergraph'], res['edge_score']
            raw_H = res['raw_H']
            bs, num, _ = raw_H.size()
            confidence, seed = res['confidence'], res['seeds']

            sampled_trans = res['sampled_trans']
            sampled_re, sampled_te = transformation_error(sampled_trans, gt_trans)
            succ = torch.where(sampled_re < config.re_thre) and torch.where(sampled_te < config.te_thre)
            recall = len(succ[0])
            correct_num += recall
            correct_ratio += recall / sampled_trans.shape[0] * 100.0
            # k = round(num * 0.1)
            # topk, _ = torch.topk(edge_score, k=k, dim=-1)  # 每个节点选出topk概率的超边，该超边包含该节点
            # a_min = torch.min(topk, dim=-1).values.unsqueeze(-1).repeat(1, 1, num)
            # W = torch.where(torch.greater_equal(edge_score, a_min), edge_score, torch.zeros_like(edge_score))
            # H = torch.where(torch.greater(W, 0), H, torch.zeros_like(H))

            score = hyperedge_loss(H, raw_H, gt_labels).detach().cpu().numpy()
            score1 = hyperedge_loss(raw_H, raw_H, gt_labels).detach().cpu().numpy()
            H_score.append(score)
            raw_H_score.append(score1)

            # evaluate raw FCGF + ransac   
            # src_pcd = make_point_cloud(src_keypts[0].detach().cpu().numpy())
            # tgt_pcd = make_point_cloud(tgt_keypts[0].detach().cpu().numpy())
            # correspondence = np.array([np.arange(src_keypts.shape[1]), np.arange(src_keypts.shape[1])])
            # correspondence = o3d.utility.Vector2iVector(correspondence.T)
            # reg_result = o3d.registration.registration_ransac_based_on_correspondence(
            #     src_pcd, tgt_pcd, correspondence,
            #     max_correspondence_distance=config.inlier_threshold,
            #     estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            #     ransac_n=3,
            #     criteria=o3d.registration.RANSACConvergenceCriteria(max_iteration=50000, max_validation=1000)
            # )
            # pred_trans = torch.eye(4)[None].to(gt_trans.device)
            # pred_trans[0, :4, :4] = torch.from_numpy(reg_result.transformation)
            # pred_labels = torch.zeros_like(gt_labels)
            # pred_labels[0, np.array(reg_result.correspondence_set)[:, 0]] = 1
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


    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 28,
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 24,
    }

    import matplotlib.pyplot as plt
    H_data = np.array(H_score)
    raw_H_data = np.array(raw_H_score)
    #plt.rc('font', family='Times New Roman')
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.letax.preample'] = r'\usepackage{}'
    width = 0.04
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.hist(raw_H_data + 0.05 - width, bins=bins + 0.05 - width, color='skyblue', edgecolor='black',
            alpha=0.7, label='initial graph', width=width, align='mid')
    ax.hist(H_data + 0.05, bins=bins + 0.05, color='orange', edgecolor='black',
            alpha=0.7, label='output graph', width=width, align='mid')
    # ax.set_title('Raw vs Processed Score Distribution')
    ax.set_xlabel('Score', font1)
    ax.set_ylabel('Frequency', font1)
    ax.set_xlim(-0.02, 1)
    ax.set_ylim(0, 500)
    ax.legend(prop=font2)
    plt.xticks(bins, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.tick_params(labelsize=24)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.tight_layout()
    plt.savefig(f'snapshot/{args.chosen_snapshot}/LoMatch_{args.descriptor}_{args.num_points}.png')
    #plt.show()
    return stats, final_poses, seed_num, seed_precision, correct_num, correct_ratio


def eval_3DMatch(model, config, args):
    dset = ThreeDLOMatchTest(root=config.root,
                            descriptor=args.descriptor,
                            inlier_threshold=config.inlier_threshold,
                            num_node=args.num_points,
                            augment_axis=0,
                            augment_rotation=0.00,
                            augment_translation=0.0,
                            )
    dloader = get_dataloader(dset, batch_size=1, num_workers=8, shuffle=False)
    os.makedirs('logs/3dlomatch', exist_ok=True)
    if os.path.isfile('logs/3dlomatch/'+args.descriptor+'.txt'):
        os.remove('logs/3dlomatch/'+args.descriptor+'.txt')
        print("File Deleted successfully")
    allpair_stats, allpair_poses, avg_seed_num, avg_seed_precision, avg_correct_num, avg_correct_ratio = eval_3DMatch_scene(model, 0, dloader, config, args)

    # benchmarking using the registration recall defined in DGR 
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Success Rate={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6] * 100:.2f}%, recall={allpair_average[7] * 100:.2f}%, f1={allpair_average[8] * 100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    print(avg_seed_num / 1781, avg_seed_precision / 1781)
    print(avg_correct_num / 1781, avg_correct_ratio / 1781)

    # benchmarking using the registration recall defined in 3DMatch to compare with Predator
    # np.save('predator.npy', allpair_poses)
    if args.descriptor == 'predator':
        benchmark_predator(allpair_poses, gt_folder='benchmarks/3DLoMatch')

    return allpair_stats


def benchmark_predator(pred_poses, gt_folder):
    scenes = sorted(os.listdir(gt_folder))
    scene_names = [os.path.join(gt_folder,ele) for ele in scenes]

    re_per_scene = defaultdict(list)
    te_per_scene = defaultdict(list)
    re_all, te_all, precision, recall = [], [], [], []
    n_valids= []

    short_names=['Kitchen','Home 1','Home 2','Hotel 1','Hotel 2','Hotel 3','Study','MIT Lab']
    logging.info(("Scene\t¦ prec.\t¦ rec.\t¦ re\t¦ te\t¦ samples\t¦"))
    
    start_ind = 0
    for idx,scene in enumerate(scene_names):
        # ground truth info
        gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
        n_valid=0
        for ele in gt_pairs:
            diff=abs(int(ele[0])-int(ele[1]))
            n_valid+=diff>1
        n_valids.append(n_valid)

        n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(scene,"gt.info"))

        # estimated info
        # est_pairs, est_traj = read_trajectory(os.path.join(est_folder,scenes[idx],'est.log'))
        est_traj = pred_poses[start_ind:start_ind + len(gt_pairs)]
        start_ind = start_ind + len(gt_pairs)

        temp_precision, temp_recall,c_flag = evaluate_registration(n_fragments, est_traj, gt_pairs, gt_pairs, gt_traj, gt_traj_cov)
        
        # Filter out the estimated rotation matrices
        ext_gt_traj = extract_corresponding_trajectors(gt_pairs,gt_pairs, gt_traj)

        re = rotation_error(torch.from_numpy(ext_gt_traj[:,0:3,0:3]), torch.from_numpy(est_traj[:,0:3,0:3])).cpu().numpy()[np.array(c_flag)==0]
        te = translation_error(torch.from_numpy(ext_gt_traj[:,0:3,3:4]), torch.from_numpy(est_traj[:,0:3,3:4])).cpu().numpy()[np.array(c_flag)==0]

        re_per_scene['mean'].append(np.mean(re))
        re_per_scene['median'].append(np.median(re))
        re_per_scene['min'].append(np.min(re))
        re_per_scene['max'].append(np.max(re))
        
        te_per_scene['mean'].append(np.mean(te))
        te_per_scene['median'].append(np.median(te))
        te_per_scene['min'].append(np.min(te))
        te_per_scene['max'].append(np.max(te))


        re_all.extend(re.reshape(-1).tolist())
        te_all.extend(te.reshape(-1).tolist())

        precision.append(temp_precision)
        recall.append(temp_recall)

        logging.info("{}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:3d}¦".format(short_names[idx], temp_precision, temp_recall, np.median(re), np.median(te), n_valid))
        # np.save(f'{est_folder}/{scenes[idx]}/flag.npy',c_flag)
    
    weighted_precision = (np.array(n_valids) * np.array(precision)).sum() / np.sum(n_valids)

    logging.info("Mean precision: {:.3f}: +- {:.3f}".format(np.mean(precision),np.std(precision)))
    logging.info("Weighted precision: {:.3f}".format(weighted_precision))

    logging.info("Mean median RRE: {:.3f}: +- {:.3f}".format(np.mean(re_per_scene['median']), np.std(re_per_scene['median'])))
    logging.info("Mean median RTE: {:.3F}: +- {:.3f}".format(np.mean(te_per_scene['median']),np.std(te_per_scene['median'])))
    

if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='HyperGCT_3DMatch_release', type=str, help='snapshot dir')
    parser.add_argument('--descriptor', default='fpfh', type=str)
    parser.add_argument('--num_points', default='all', type=str)
    parser.add_argument('--use_icp', default=False, type=str2bool)
    args = parser.parse_args()
    print(args.chosen_snapshot)

    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)
    config.inlier_threshold = 0.1
    config.re_thre = 15
    config.te_thre = 30
    # change the dataset path here
    # config.root = ''
    # if args.descriptor == 'predator':
    #    config.root = ''
    
    log_filename = f'logs/3DLoMatch_{args.chosen_snapshot}-{args.descriptor}-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))   

    config.mode = "test"
    from models.mymodel import MethodName
    model = MethodName(config)

    miss = model.load_state_dict(torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best.pkl'), strict=True)
    print(miss)
    model.eval()

    # evaluate on the test set
    stats = eval_3DMatch(model.cuda(), config, args)
