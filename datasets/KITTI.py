import os
import torch.utils.data as data
from utils.pointcloud import make_point_cloud, estimate_normal_gpu
from utils.SE3 import *

class KITTIDataset(data.Dataset):
    def __init__(self,
                root,
                split='train',
                descriptor='fcgf',
                in_dim=6,
                inlier_threshold=0.60,
                num_node=5000,
                use_mutual=True,
                downsample=0.30,
                augment_axis=0,
                augment_rotation=1.0,
                augment_translation=0.01,
                ):
        self.root = root
        self.split = split
        self.descriptor = descriptor
        #assert descriptor in ['fcgf', 'fpfh']
        self.in_dim = in_dim
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

        # containers
        self.ids_list = []

        for filename in os.listdir(f"{self.root}/{descriptor}_{split}/"):
            self.ids_list.append(os.path.join(f"{self.root}/{descriptor}_{split}/", filename))

        # self.ids_list = sorted(self.ids_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    def __getitem__(self, index):
        # load meta data
        filename = self.ids_list[index]
        data = np.load(filename)
        src_keypts = data['xyz0']
        tgt_keypts = data['xyz1']
        src_features = data['features0']
        tgt_features = data['features1']
        src_normal = data['normal0']
        tgt_normal = data['normal1']
        if self.descriptor == 'fpfh':
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        orig_trans = data['gt_trans']
        # data augmentation
        if self.split == 'train':
            src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.05
            tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.05
            aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
            aug_T = translation_matrix(self.augment_translation)
            aug_trans = integrate_trans(aug_R, aug_T)
            tgt_keypts = transform(tgt_keypts, aug_trans)
            gt_trans = concatenate(aug_trans, orig_trans)
        else:
            gt_trans = orig_trans
        
        
        # estimate normal
        # src_pcd = make_point_cloud(src_keypts)
        # tgt_pcd = make_point_cloud(tgt_keypts)
        # src_normal = estimate_normal_gpu(src_pcd, radius=1.2)
        # tgt_normal = estimate_normal_gpu(tgt_pcd, radius=1.2)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]

        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            if self.num_node < N_tgt:
                tgt_sel_ind = np.random.choice(N_tgt, self.num_node, replace=False)
            else:
                tgt_sel_ind = np.arange(N_tgt)

            if self.num_node < N_src:
                src_sel_ind = np.random.choice(N_src, self.num_node, replace=False)
            else:
                src_sel_ind = np.arange(N_src)
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]
        
        src_normal = src_normal[src_sel_ind, :]
        tgt_normal = tgt_normal[tgt_sel_ind, :]

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
        
        # test only
        if self.split == 'test':
            probs = (score / score.sum()).flatten()
            sel_ind = np.random.choice(corr.shape[0], size=corr0.shape[0], replace=False, p=probs)
            corr = corr[sel_ind, :]
        
        if len(corr) < 10:
            # skip pairs with too few correspondences.
            return self.__getitem__(int(np.random.choice(self.__len__(), 1)))

        # compute the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int32)
        
        # prepare input to the network
        input_src_keypts = src_keypts[corr[:, 0]]
        input_tgt_keypts = tgt_keypts[corr[:, 1]]

        input_src_normal = src_normal[corr[:, 0]]
        input_tgt_normal = tgt_normal[corr[:, 1]]
        
        corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
        # move the center of each point cloud to (0,0,0).
        corr_pos = corr_pos - corr_pos.mean(0)
 

        filename = os.path.basename(filename)
        filename = filename.split('.')[0]

        return corr_pos.astype(np.float32), \
            input_src_keypts.astype(np.float32), \
            input_tgt_keypts.astype(np.float32), \
            input_src_normal.astype(np.float32), \
            input_tgt_normal.astype(np.float32), \
            gt_trans.astype(np.float32), \
            labels.astype(np.float32),

    def __len__(self):
        return len(self.ids_list)


def extract_ids(path):
    # 提取 id1, id2, id3
    filename = path.split('/')[-1].split('.')[0]
    parts = filename.split('-')
    id1 = int(parts[0][5:])  # drive{id1}
    pair_part = parts[1].split('_')[0]
    id2 = int(pair_part[4:])  # pair{id2}
    id3 = int(parts[1].split('_')[1])  # {id3}
    return (id1, id2, id3)

class KITTIDatasetTest(data.Dataset):
    def __init__(self,
                 root,
                 split='test',
                 descriptor='fcgf',
                 inlier_threshold=0.60,
                 num_node=5000,
                 use_mutual=True,
                 downsample=0.30,
                 augment_axis=0,
                 augment_rotation=1.0,
                 augment_translation=0.01,
                 args=None
                 ):
        self.root = root
        self.split = split
        self.descriptor = descriptor
        # assert descriptor in ['fcgf', 'fpfh']
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

        # containers
        self.ids_list = []
        if args.dataset == "KITTI_10m":
            for filename in os.listdir(f"{self.root}/{descriptor}_{split}/"):
                self.ids_list.append(os.path.join(f"{self.root}/{descriptor}_{split}/", filename))
        elif args.dataset == "KITTI_LC":
            for filename in os.listdir(f"{self.root}/lc_{descriptor}_test_{args.range}/"):
                self.ids_list.append(os.path.join(f"{self.root}/lc_{descriptor}_test_{args.range}/", filename))
        else:
            raise NotImplementedError
        #self.ids_list = sorted(self.ids_list, key=extract_ids)

    def __getitem__(self, index):
        # load meta data
        filename = self.ids_list[index]
        data = np.load(filename)
        src_keypts = data['xyz0']
        tgt_keypts = data['xyz1']
        src_features = data['features0']
        tgt_features = data['features1']
        src_normal = data['normal0']
        tgt_normal = data['normal1']
        if self.descriptor == 'fpfh':
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        orig_trans = data['gt_trans']
        gt_trans = orig_trans

        # estimate normal
        # src_pcd = make_point_cloud(src_keypts)
        # tgt_pcd = make_point_cloud(tgt_keypts)
        # src_normal = estimate_normal_gpu(src_pcd, radius=1.2)
        # tgt_normal = estimate_normal_gpu(tgt_pcd, radius=1.2)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]

        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            if self.num_node < N_tgt:
                tgt_sel_ind = np.random.choice(N_tgt, self.num_node, replace=False)
            else:
                tgt_sel_ind = np.arange(N_tgt)

            if self.num_node < N_src:
                src_sel_ind = np.random.choice(N_src, self.num_node, replace=False)
            else:
                src_sel_ind = np.arange(N_src)
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]

        src_normal = src_normal[src_sel_ind, :]
        tgt_normal = tgt_normal[tgt_sel_ind, :]

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

        # test only
        probs = (score / score.sum()).flatten()
        sel_ind = np.random.choice(corr.shape[0], size=corr0.shape[0], replace=False, p=probs)
        corr = corr[sel_ind, :]


        # compute the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int32)

        # prepare input to the network
        input_src_keypts = src_keypts[corr[:, 0]]
        input_tgt_keypts = tgt_keypts[corr[:, 1]]

        input_src_normal = src_normal[corr[:, 0]]
        input_tgt_normal = tgt_normal[corr[:, 1]]

        corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
        # move the center of each point cloud to (0,0,0).
        corr_pos = corr_pos - corr_pos.mean(0)

        filename = os.path.basename(filename)
        filename = filename.split('.')[0]

        return corr_pos.astype(np.float32), \
            input_src_keypts.astype(np.float32), \
            input_tgt_keypts.astype(np.float32), \
            input_src_normal.astype(np.float32), \
            input_tgt_normal.astype(np.float32), \
            gt_trans.astype(np.float32), \
            labels.astype(np.float32),\
            filename

    def __len__(self):
        return len(self.ids_list)



if __name__ == "__main__":
    dset = KITTIDataset(
                    root='/data/KITTI/',
                    split='train',
                    descriptor='feat',
                    num_node=5000,
                    use_mutual=False,
                    augment_axis=0,
                    augment_rotation=0,
                    augment_translation=0.00
                    )
    print(len(dset))
    for i in range(dset.__len__()):
        ret_dict = dset[i]
