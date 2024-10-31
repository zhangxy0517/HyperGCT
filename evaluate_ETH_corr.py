import sys
from easydict import EasyDict as edict
import json
sys.path.append('/')
import open3d
import numpy as np
import time
from sklearn.neighbors import KDTree
import glob
import math
import os

import torch
from utils.pointcloud import make_point_cloud, estimate_normal_gpu
from utils.SE3 import *


def get_pcd(pcdpath, filename):
    return open3d.io.read_point_cloud(os.path.join(pcdpath, filename + '.ply'))


def get_ETH_keypts(pcd, keyptspath, filename):
    pts = np.array(pcd.points)
    key_ind = np.loadtxt(os.path.join(keyptspath, filename + '_Keypoints.txt'), dtype=np.int32)
    keypts = pts[key_ind]
    return keypts


def get_desc(descpath, filename, desc_name):
    if desc_name == '3dmatch':
        desc = np.fromfile(os.path.join(descpath, filename + '.desc.3dmatch.bin'), dtype=np.float32)
        num_desc = int(desc[0])
        desc_size = int(desc[1])
        desc = desc[2:].reshape([num_desc, desc_size])
    elif desc_name in ['SpinNet', 'fpfh', 'fcgf']:
        desc = np.load(os.path.join(descpath, filename + f'.desc.{desc_name}.bin.npy'))
    else:
        print("No such descriptor")
        exit(-1)
    return desc


def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result


def calculate_M(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 512]
    """

    kdtree_s = KDTree(target_desc)
    sourceNNdis, sourceNNidx = kdtree_s.query(source_desc, 1)
    kdtree_t = KDTree(source_desc)
    targetNNdis, targetNNidx = kdtree_t.query(target_desc, 1)
    result = []
    for i in range(len(sourceNNidx)):
        if targetNNidx[sourceNNidx[i]] == i:
            result.append([i, sourceNNidx[i][0]])
    return np.array(result)

def calculate_M1(src_desc, tgt_desc):
    # construct the correspondence set by mutual nn in feature space.
    distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)

    relax_src_dis, relax_src_idx = torch.topk(torch.from_numpy(distance), k=2, dim=1, largest=False)
    source_idx = relax_src_idx[:, 0].numpy()
    # source_idx = np.argmin(distance, axis=1)  # for each row save the index of minimun
    corr0 = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
    score0 = (relax_src_dis[:, 0] / relax_src_dis[:, 1]).numpy()

    relax_tgt_dis, relax_tgt_idx = torch.topk(torch.from_numpy(distance), k=2, dim=0, largest=False)
    relax_tgt_dis = relax_tgt_dis.T
    relax_tgt_idx = relax_tgt_idx.T
    target_idx = relax_tgt_idx[:, 0].numpy()
    # target_idx = np.argmin(distance, axis=0)
    corr1 = np.concatenate([target_idx[:, None], np.arange(target_idx.shape[0])[:, None]], axis=-1)
    score1 = (relax_tgt_dis[:, 0] / relax_tgt_dis[:, 1]).numpy()

    mask = source_idx[target_idx] != np.arange(target_idx.shape[0])  # 找出corr1不重复的匹配
    corr = np.concatenate([corr0, corr1[mask]], axis=0)
    score = np.concatenate([score0, score1[mask]], axis=0)

    probs = (score / score.sum()).flatten()
    sel_ind = np.random.choice(corr.shape[0], size=corr0.shape[0], replace=False, p=probs)
    corr = corr[sel_ind, :]
    return corr

# def decompose_trans(trans):
#     """
#     Decompose SE3 transformations into R and t, support torch.Tensor and np.ndarry.
#     Input
#         - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
#     Output
#         - R: [3, 3] or [bs, 3, 3], rotation matrix
#         - t: [3, 1] or [bs, 3, 1], translation matrix
#     """
#     if len(trans.shape) == 3:
#         return trans[:, :3, :3], trans[:, :3, 3:4]
#     else:
#         return trans[:3, :3], trans[:3, 3:4]


def register2Fragments(model, id1, id2, keyptspath, descpath, resultpath, desc_name, config):
    cloud_bin_s = f'Hokuyo_{id1}'
    cloud_bin_t = f'Hokuyo_{id2}'
    write_file = f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'
    if desc_name == 'SpinNet':
        pcd_s = get_pcd(pcdpath, cloud_bin_s)
        source_keypts = get_ETH_keypts(pcd_s, keyptspath, cloud_bin_s)
        pcd_t = get_pcd(pcdpath, cloud_bin_t)
        target_keypts = get_ETH_keypts(pcd_t, keyptspath, cloud_bin_t)
        src_pcd = make_point_cloud(source_keypts)
        tgt_pcd = make_point_cloud(target_keypts)
        source_normal = estimate_normal_gpu(src_pcd, radius=0.2)
        target_normal = estimate_normal_gpu(tgt_pcd, radius=0.2)
        source_keypts = source_keypts[-num_keypoints:, :]
        target_keypts = target_keypts[-num_keypoints:, :]
        source_normal = source_normal[-num_keypoints:, :]
        target_normal = target_normal[-num_keypoints:, :]

        source_desc = get_desc(descpath, cloud_bin_s, desc_name=desc_name)
        target_desc = get_desc(descpath, cloud_bin_t, desc_name=desc_name)
        source_desc = np.nan_to_num(source_desc)
        target_desc = np.nan_to_num(target_desc)
        source_desc = source_desc[-num_keypoints:, :]
        target_desc = target_desc[-num_keypoints:, :]
        source_desc = source_desc / (np.linalg.norm(source_desc, axis=1, keepdims=True) + 1e-6)
        target_desc = target_desc / (np.linalg.norm(target_desc, axis=1, keepdims=True) + 1e-6)

    else:
        src_data = np.load(f"{descpath}/{cloud_bin_s}.desc.{desc_name}.bin.npz")
        tgt_data = np.load(f"{descpath}/{cloud_bin_t}.desc.{desc_name}.bin.npz")
        source_keypts = src_data['xyz']
        target_keypts = tgt_data['xyz']
        source_desc = src_data['feature']
        target_desc = tgt_data['feature']
        if desc_name == 'fpfh':
            source_desc = source_desc / (np.linalg.norm(source_desc, axis=1, keepdims=True) + 1e-6)
            target_desc = target_desc / (np.linalg.norm(target_desc, axis=1, keepdims=True) + 1e-6)
        src_pcd = make_point_cloud(source_keypts)
        tgt_pcd = make_point_cloud(target_keypts)
        source_normal = estimate_normal_gpu(src_pcd, radius=0.2)
        target_normal = estimate_normal_gpu(tgt_pcd, radius=0.2)

        N_src = source_desc.shape[0]
        N_tgt = target_desc.shape[0]
        src_sel_ind = np.random.choice(N_src, num_keypoints, replace=False)
        tgt_sel_ind = np.random.choice(N_tgt, num_keypoints, replace=False)
        source_desc = source_desc[src_sel_ind, :]
        target_desc = target_desc[tgt_sel_ind, :]
        source_keypts = source_keypts[src_sel_ind, :]
        target_keypts = target_keypts[tgt_sel_ind, :]
        source_normal = source_normal[src_sel_ind, :]
        target_normal = target_normal[tgt_sel_ind, :]

    key = f'{cloud_bin_s.split("_")[-1]}_{cloud_bin_t.split("_")[-1]}'
    reg = 0
    if key not in gtLog.keys():
        num_inliers = 0
        inlier_ratio = 0
        gt_flag = 0
    else:
        corr = calculate_M1(source_desc, target_desc)
        gtTrans = gtLog[key]
        frag1 = source_keypts[corr[:, 0]]
        frag2_pc = open3d.geometry.PointCloud()
        frag2_pc.points = open3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
        frag2_pc.transform(gtTrans)
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        labels = (distance < config.inlier_threshold).astype(np.int32)
        num_inliers = np.sum(labels)
        inlier_ratio = num_inliers / len(distance)
        gt_flag = 1
        source_keypts = source_keypts[corr[:, 0]]
        target_keypts = target_keypts[corr[:, 1]]

        source_normal = source_normal[corr[:, 0]]
        target_normal = target_normal[corr[:, 1]]

        corr_pos = np.concatenate([source_keypts, target_keypts], axis=-1)
        corr_pos = corr_pos - corr_pos.mean(0)

        corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels = \
            (torch.from_numpy(corr_pos)[None].cuda(), torch.from_numpy(source_keypts)[None].cuda(), torch.from_numpy(target_keypts)[None].cuda(),
             torch.from_numpy(source_normal)[None].cuda(), torch.from_numpy(target_normal)[None].cuda(), torch.from_numpy(gtTrans)[None].cuda(),
             torch.from_numpy(labels)[None].cuda())
        data = {
            'corr_pos': corr_pos,
            'src_keypts': src_keypts,
            'tgt_keypts': tgt_keypts,
            'src_normal': src_normal,
            'tgt_normal': tgt_normal,
            'labels': gt_labels,
            'testing': True,
        }

        res = model(data)
        pred_trans, pred_labels = res['final_trans'].cpu().numpy(), res['final_labels'].cpu().numpy()

        # write the transformation matrix into .log file for evaluation.
        with open(os.path.join(logpath, f'{desc_name}.log'), 'a+') as f:
            trans = pred_trans[0]
            trans = np.linalg.inv(trans)
            gt_R, gt_T = decompose_trans(gtTrans)
            R, T = decompose_trans(trans)
            re = math.acos(np.clip((np.trace(R.T @ gt_R) - 1) / 2.0, a_min=-1, a_max=1))
            te = np.sqrt(np.sum((T - gt_T) ** 2))
            re = re * 180 / np.pi
            te = te * 100

            if te < config.te_thre and re < config.re_thre:
                reg = 1
            s1 = f'{id1}\t {id2}\t {reg}\t {re}\t {te}\n'
            f.write(s1)
            f.write(f"{trans[0, 0]}\t {trans[0, 1]}\t {trans[0, 2]}\t {trans[0, 3]}\t \n")
            f.write(f"{trans[1, 0]}\t {trans[1, 1]}\t {trans[1, 2]}\t {trans[1, 3]}\t \n")
            f.write(f"{trans[2, 0]}\t {trans[2, 1]}\t {trans[2, 2]}\t {trans[2, 3]}\t \n")
            f.write(f"{trans[3, 0]}\t {trans[3, 1]}\t {trans[3, 2]}\t {trans[3, 3]}\t \n")

    s = f"{cloud_bin_s}\t{cloud_bin_t}\t{num_inliers}\t{inlier_ratio:.8f}\t{gt_flag}"
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'w+') as f:
        f.write(s)
    return num_inliers, inlier_ratio, gt_flag, reg


def read_register_result(id1, id2):
    cloud_bin_s = f'Hokuyo_{id1}'
    cloud_bin_t = f'Hokuyo_{id2}'
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'r') as f:
        content = f.readlines()
    nums = content[0].replace("\n", "").split("\t")[2:5]
    return nums


if __name__ == '__main__':
    scene_list = [
        'gazebo_summer',
        'gazebo_winter',
        'wood_autmn',
        'wood_summer',
    ]

    inliers_list = []
    recall_list = []

    chosen_snapshot = 'HyperGCT_3DMatch_release'
    config_path = f'snapshot/{chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    desc_name = 'fcgf'
    num_keypoints = 5000
    config.inlier_threshold = 0.1
    config.re_thre, config.te_thre = 15, 30
    config.mode = 'test'
    # change the dataset path here
    root = '/data/zxy/ETH'

    from models.mymodel import MethodName

    model = MethodName(config)
    miss = model.load_state_dict(torch.load(f'snapshot/{chosen_snapshot}/models/model_best.pkl'), strict=True)
    print(miss)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    for scene in scene_list:
        pcdpath = f"{root}/data/{scene}/"
        interpath = f"{root}/data/{scene}/01_Keypoints/"
        gtpath = f'{root}/data/{scene}/'
        keyptspath = interpath  # os.path.join(interpath, "keypoints/")
        descpath = os.path.join(f"{root}/descriptor", f"SpinNet_desc_{desc_name}/{scene}")
        logpath = f"logs/eth/{scene}-evaluation"
        gtLog = loadlog(gtpath)
        resultpath = os.path.join("./ETH", f"SpinNet_result_{desc_name}/{scene}")
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        if os.path.isfile(os.path.join(logpath, f'{desc_name}.log')):
            os.remove(os.path.join(logpath, f'{desc_name}.log'))
            print("File Deleted successfully")

        # register each pair
        fragments = glob.glob(pcdpath + '*.ply')
        num_frag = len(fragments)
        success_reg = 0
        total = 0
        print(f"Start Evaluating Scene {scene}")
        start_time = time.time()
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                num_inliers, inlier_ratio, gt_flag, reg = register2Fragments(model, id1, id2, keyptspath, descpath,resultpath,
                                                                        desc_name, config)
                success_reg += reg
                total += gt_flag
        print(f"Finish Evaluation, time: {time.time() - start_time:.2f}s")
        print(str(success_reg) + ' ' + str(total) + ' ' + str(float(success_reg / total) * 100))

        # evaluate
        result = []
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                line = read_register_result(id1, id2)
                result.append([int(line[0]), float(line[1]), int(line[2])])
        result = np.array(result)
        indices_results = np.sum(result[:, 2] == 1)
        correct_match = np.sum(result[:, 1] > 0.05)
        recall = float(correct_match / indices_results) * 100
        print(f"Correct Match {correct_match}, ground truth Match {indices_results}")
        print(f"Recall {recall}%")
        ave_num_inliers = np.sum(np.where(result[:, 1] > 0.05, result[:, 0], np.zeros(result.shape[0]))) / correct_match
        print(f"Average Num Inliners: {ave_num_inliers}")
        recall_list.append(recall)
        inliers_list.append(ave_num_inliers)
    print(recall_list)
    average_recall = sum(recall_list) / len(recall_list)
    print(f"All 8 scene, average recall: {average_recall}%")
    average_inliers = sum(inliers_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers: {average_inliers}")