import os
import open3d as o3d
import numpy as np
from utils.pointcloud import make_point_cloud
import glob
import MinkowskiEngine as ME
from tqdm import tqdm


def process_3dmatch(voxel_size=0.05):
    root = "../data/3DMatch/threedmatch/"
    save_path = "../data/3DMatch/threedmatch_feat/"
    pcd_list = os.listdir(root)
    for pcd_path in pcd_list:
        if pcd_path.endswith('.npz') is False:
            continue
        full_path = os.path.join(root, pcd_path)
        data = np.load(full_path)
        pts = data['pcd']
        if pts.shape[0] == 0:
            print(f"{full_path} error: do not have any points.")
            continue
        orig_pcd = make_point_cloud(pts)
        # voxel downsample 
        pcd = orig_pcd.voxel_down_sample(voxel_size=voxel_size)

        # estimate the normals and compute fpfh descriptor
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
        fpfh_np = np.array(fpfh.data).T

        # save the data for training.
        np.savez_compressed(
            os.path.join(save_path, pcd_path.replace('.npz', '_fpfh.npz')),
            points=np.array(orig_pcd.points).astype(np.float32),
            xyz=np.array(pcd.points).astype(np.float32),
            feature=fpfh_np.astype(np.float32),
        )
        print(full_path, fpfh_np.shape)


def process_3dmatch_test(voxel_size=0.05):
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
    for scene in scene_list:
        scene_path = os.path.join("/ssd2/xuyang/3DMatch/fragments/", scene)
        pcd_list = os.listdir(scene_path)
        for pcd_path in pcd_list:
            if not pcd_path.endswith('.ply'):
                continue
            full_path = os.path.join(scene_path, pcd_path)
            orig_pcd = o3d.io.read_point_cloud(full_path)
            # voxel downsample 
            pcd = orig_pcd.voxel_down_sample(voxel_size=voxel_size)

            # estimate the normals and compute fpfh descriptor 
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 5, max_nn=100))
            fpfh_np = np.array(fpfh.data).T

            # save the data for training.
            np.savez_compressed(
                full_path.replace('.ply', '_fpfh'),
                points=np.array(orig_pcd.points).astype(np.float32),
                xyz=np.array(pcd.points).astype(np.float32),
                feature=fpfh_np.astype(np.float32),
            )
            print(full_path, fpfh_np.shape)


def process_redwood(voxel_size=0.05):
    scene_list = [
        'livingroom1-simulated',
        'livingroom2-simulated',
        'office1-simulated',
        'office2-simulated'
    ]
    for scene in scene_list:
        scene_path = os.path.join("../data/Augmented_ICL-NUIM/", scene + '/fragments')
        pcd_list = os.listdir(scene_path)
        for pcd_path in pcd_list:
            if not pcd_path.endswith('.ply'):
                continue
            full_path = os.path.join(scene_path, pcd_path)
            orig_pcd = o3d.io.read_point_cloud(full_path)
            # voxel downsample 
            pcd = orig_pcd.voxel_down_sample(voxel_size=voxel_size)

            # estimate the normals and compute fpfh descriptor
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 5, max_nn=100))
            fpfh_np = np.array(fpfh.data).T

            # save the data for training.
            np.savez_compressed(
                full_path.replace('.ply', '_fpfh'),
                points=np.array(orig_pcd.points).astype(np.float32),
                xyz=np.array(pcd.points).astype(np.float32),
                feature=fpfh_np.astype(np.float32),
            )
            print(full_path, fpfh_np.shape)


kitti_cache = {}
kitti_icp_cache = {}


#KITTI数据集
def process_kitti(voxel_size=0.30, split='train'):
    def odometry_to_positions(odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def get_video_odometry(root, drive, indices=None, ext='.txt', return_all=False):
        data_path = root + '/poses/%02d.txt' % drive
        if data_path not in kitti_cache:
            kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return kitti_cache[data_path]
        else:
            return kitti_cache[data_path][indices]

    def _get_velodyne_fn(root, drive, t):
        fname = root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def apply_transform(pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    MIN_DIST = 10
    root = '/data/zxy/KITTI'
    R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T

    subset_names = open(f'misc/split/{split}_kitti.txt').read().split()
    files = []
    for dirname in subset_names:
        drive_id = int(dirname)
        fnames = glob.glob(root + '/sequences/%02d/velodyne/*.bin' % drive_id)

        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = get_video_odometry(root, drive_id, return_all=True)
        all_pos = np.array([odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
        pdist = np.sqrt(pdist.sum(-1))
        more_than_10 = pdist > MIN_DIST
        curr_time = inames[0]
        while curr_time in inames:
            next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
            if len(next_time) == 0:
                curr_time += 1
            else:
                next_time = next_time[0] + curr_time - 1

            if next_time in inames:
                files.append((drive_id, curr_time, next_time))
                curr_time = next_time + 1
        # Remove problematic sequence
        for item in [
            (8, 15, 58),
        ]:
            if item in files:
                files.pop(files.index(item))

    # begin extracting features
    for idx in tqdm(range(len(files)), ncols=50):
        drive = files[idx][0]
        t0, t1 = files[idx][1], files[idx][2]
        all_odometry = get_video_odometry(root, drive, [t0, t1])
        positions = [odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = _get_velodyne_fn(root, drive, t0)
        fname1 = _get_velodyne_fn(root, drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = root + '/fpfh_icp/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                # work on the downsampled xyzs, 0.05m == 5cm
                _, sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
                _, sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

                M = (velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                     @ np.linalg.inv(velo2cam)).T
                xyz0_t = apply_transform(xyz0[sel0], M)
                pcd0 = make_point_cloud(xyz0_t)
                pcd1 = make_point_cloud(xyz1[sel1])
                reg = o3d.pipelines.registration.registration_icp(
                    pcd0, pcd1, 0.2, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
                M2 = M @ reg.transformation
                # o3d.draw_geometries([pcd0, pcd1])
                # write to a file
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]

        pcd0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz0))
        # voxel downsample
        pcd0 = pcd0.voxel_down_sample(voxel_size=voxel_size)

        # estimate the normals and compute fpfh descriptor
        pcd0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh0 = o3d.pipelines.registration.compute_fpfh_feature(pcd0, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
        fpfh_np0 = np.array(fpfh0.data).T

        normals0 = np.asarray(pcd0.normals)

        pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz1))
        # voxel downsample
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

        # estimate the normals and compute fpfh descriptor
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(pcd1, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
        fpfh_np1 = np.array(fpfh1.data).T

        normals1 = np.asarray(pcd1.normals)

        # save the data for training.
        filename = f"{root}/fpfh_{split}/drive{drive}-pair{t0}_{t1}"
        np.savez_compressed(
            filename,
            xyz0=np.asarray(pcd0.points).astype(np.float32),
            xyz1=np.asarray(pcd1.points).astype(np.float32),
            features0=fpfh_np0.astype(np.float32),
            features1=fpfh_np1.astype(np.float32),
            normal0=normals0.astype(np.float32),
            normal1=normals1.astype(np.float32),
            gt_trans=M2.astype(np.float32)
        )


def process_lokitti(voxel_size=0.30):
    def odometry_to_positions(odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def get_video_odometry(root, drive, indices=None, ext='.txt', return_all=False):
        data_path = root + '/poses/%02d.txt' % drive
        if data_path not in kitti_cache:
            kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return kitti_cache[data_path]
        else:
            return kitti_cache[data_path][indices]

    def _get_velodyne_fn(root, drive, t):
        fname = root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def apply_transform(pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    MIN_DIST = 10
    root = '/data/zxy/KITTI'
    R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T

    # subset_names = open(f'misc/split/{split}_kitti.txt').read().split()
    # files = []
    # for dirname in subset_names:
    #     drive_id = int(dirname)
    #     fnames = glob.glob(root + '/sequences/%02d/velodyne/*.bin' % drive_id)
    #
    #     assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
    #     inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
    #
    #     all_odo = get_video_odometry(root, drive_id, return_all=True)
    #     all_pos = np.array([odometry_to_positions(odo) for odo in all_odo])
    #     Ts = all_pos[:, :3, 3]
    #     pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
    #     pdist = np.sqrt(pdist.sum(-1))
    #     more_than_10 = pdist > MIN_DIST
    #     curr_time = inames[0]
    #     while curr_time in inames:
    #         next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
    #         if len(next_time) == 0:
    #             curr_time += 1
    #         else:
    #             next_time = next_time[0] + curr_time - 1
    #
    #         if next_time in inames:
    #             files.append((drive_id, curr_time, next_time))
    #             curr_time = next_time + 1
    #     # Remove problematic sequence
    #     for item in [
    #         (8, 15, 58),
    #     ]:
    #         if item in files:
    #             files.pop(files.index(item))

    files = np.load('misc/file_LoKITTI_50.npy')
    # begin extracting features
    for idx in tqdm(range(len(files)), ncols=50):
        drive = files[idx][0]
        t0, t1 = files[idx][1], files[idx][2]
        all_odometry = get_video_odometry(root, drive, [t0, t1])
        positions = [odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = _get_velodyne_fn(root, drive, t0)
        fname1 = _get_velodyne_fn(root, drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = root + '/lo_fpfh_icp/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                # work on the downsampled xyzs, 0.05m == 5cm
                _, sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
                _, sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

                M = (velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                     @ np.linalg.inv(velo2cam)).T
                xyz0_t = apply_transform(xyz0[sel0], M)
                pcd0 = make_point_cloud(xyz0_t)
                pcd1 = make_point_cloud(xyz1[sel1])
                reg = o3d.pipelines.registration.registration_icp(
                    pcd0, pcd1, 0.2, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
                M2 = M @ reg.transformation
                # o3d.draw_geometries([pcd0, pcd1])
                # write to a file
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]

        pcd0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz0))
        # voxel downsample
        pcd0 = pcd0.voxel_down_sample(voxel_size=voxel_size)

        # estimate the normals and compute fpfh descriptor
        pcd0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh0 = o3d.pipelines.registration.compute_fpfh_feature(pcd0, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
        fpfh_np0 = np.array(fpfh0.data).T

        normals0 = np.asarray(pcd0.normals)

        pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz1))
        # voxel downsample
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

        # estimate the normals and compute fpfh descriptor
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(pcd1, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
        fpfh_np1 = np.array(fpfh1.data).T

        normals1 = np.asarray(pcd1.normals)

        # save the data for training.
        filename = f"{root}/lo_fpfh_test/drive{drive}-pair{t0}_{t1}"
        np.savez_compressed(
            filename,
            xyz0=np.asarray(pcd0.points).astype(np.float32),
            xyz1=np.asarray(pcd1.points).astype(np.float32),
            features0=fpfh_np0.astype(np.float32),
            features1=fpfh_np1.astype(np.float32),
            normal0=normals0.astype(np.float32),
            normal1=normals1.astype(np.float32),
            gt_trans=M2.astype(np.float32)
        )

def process_kittilc(voxel_size=0.30, test_file=None):
    def read_test_file(test_file):
        test_pairs = []
        for row, line in enumerate(open(test_file)):
            if row == 0:
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split()
            seq, i, seq_db, j = [int(x) for x in line[:4]]
            tf = np.array([float(x) for x in line[4:20]]).reshape(4, 4)
            overlap = 1.0
            if len(line) > 20:
                overlap = float(line[20])
            test_pairs.append((seq, i, seq_db, j, tf, overlap))
        return test_pairs

    def _get_velodyne_fn(root, drive, t):
        fname = root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    root = '/data/zxy/KITTI'
    files = read_test_file(test_file)
    for seq, i, seq_db, j, tf_gt, overlap in tqdm(files, ncols=50):
        fname0 = _get_velodyne_fn(root, seq, i)
        fname1 = _get_velodyne_fn(root, seq_db, j)
        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]
        M2 = tf_gt

        pcd0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz0))
        # voxel downsample
        pcd0 = pcd0.voxel_down_sample(voxel_size=voxel_size)

        # estimate the normals and compute fpfh descriptor
        pcd0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh0 = o3d.pipelines.registration.compute_fpfh_feature(pcd0, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
        fpfh_np0 = np.array(fpfh0.data).T

        normals0 = np.asarray(pcd0.normals)

        pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz1))
        # voxel downsample
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

        # estimate the normals and compute fpfh descriptor
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(pcd1, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
        fpfh_np1 = np.array(fpfh1.data).T

        normals1 = np.asarray(pcd1.normals)

        dis = test_file.split('/')[-1].split('.')[0]
        filename = f"{root}/lc_fpfh_{dis}/drive{seq}-pair{i}_{j}"
        np.savez_compressed(
            filename,
            xyz0=np.asarray(pcd0.points).astype(np.float32),
            xyz1=np.asarray(pcd1.points).astype(np.float32),
            features0=fpfh_np0.astype(np.float32),
            features1=fpfh_np1.astype(np.float32),
            normal0=normals0.astype(np.float32),
            normal1=normals1.astype(np.float32),
            gt_trans=M2.astype(np.float32)
        )

if __name__ == '__main__':
    #process_3dmatch(voxel_size=0.05)
    # process_3dmatch_test(voxel_size=0.05)
    #process_redwood(voxel_size=0.05)
    #process_lokitti(voxel_size=0.30)
    process_kittilc(voxel_size=0.30, test_file='misc/kitti_lc/test_20_30.txt')
