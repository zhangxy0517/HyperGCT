import torch
import open3d as o3d
import numpy as np

def make_point_cloud(pts):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd 

def make_feature(data, dim, npts):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    feature = o3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.astype('d').transpose()
    return feature

def estimate_normal(pcd, radius=0.06, max_nn=30):
    # cpu version
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.array(pcd.normals)

def estimate_normal_gpu(pcd, radius=0.06, max_nn=30, rank=0):
    # gpu version
    device = o3d.core.Device(f'cuda:{rank}')
    cloud = o3d.t.geometry.PointCloud(device)
    cloud.point["positions"] = o3d.core.Tensor(np.asarray(pcd.points), o3d.core.Dtype.Float32, device)
    o3d.t.geometry.PointCloud.estimate_normals(cloud, radius=radius, max_nn=max_nn)
    location = o3d.core.Tensor([0., 0., 0.], o3d.core.Dtype.Float32, device)
    o3d.t.geometry.PointCloud.orient_normals_towards_camera_location(cloud,
                                                                    camera_location=location)  # 让法向量的方向都指向给定的相机位置
    normals = cloud.point['normals'].cpu().numpy()
    return normals
