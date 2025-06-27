# HyperGCT: A Dynamic Hyper-GNN-Learned Geometric Constraint for 3D Registration (ICCV 2025)
Source code of [HyperGCT](). 

## Introduction
Geometric constraints between feature matches are critical in 3D point cloud registration problems. Existing approaches typically model unordered matches as a consistency graph and sample consistent matches to generate hypotheses. However, explicit graph construction introduces noise, posing great challenges for handcrafted geometric constraints to render consistency. To overcome this, we propose HyperGCT, a flexible dynamic **Hyper**-**G**NN-learned geometric **C**onstrain**T** that leverages high-order consistency among 3D correspondences. To our knowledge, HyperGCT is the first method that mines robust geometric constraints from dynamic hypergraphs for 3D registration. By dynamically optimizing the hypergraph through vertex and edge feature aggregation, HyperGCT effectively captures the correlations among correspondences, leading to accurate hypothesis generation. Extensive experiments on 3DMatch, 3DLoMatch, KITTI-LC, and ETH show that HyperGCT achieves state-of-the-art performance. Furthermore, HyperGCT is robust to graph noise, demonstrating a significant advantage in terms of generalization. 

## Requirements

CUDA 11.8 and conda should be installed first, then you may configure HyperGCT as:

    conda env create -f environment.yml
    conda activate HyperGCT

## Pretrained Model

We provide the pre-trained model of 3DMatch in `snapshot/HyperGCT_3DMatch_release` and KITTI in `snapshot/HyperGCT_KITTI_release`.

## Data Preparation



## Instructions to training and testing

### 3DMatch

The training and testing on 3DMatch dataset can be done by running
```bash
python train_3dmatch.py

python test_3DMatch.py --chosen_snapshot [exp_id] --descriptor FCGF --use_icp False
```
where the `exp_id` should be replaced by the snapshot folder name for testing (e.g. `HyperGCT_3DMatch_release`).  The testing results will be saved in `logs/`. The training config can be changed in `config.py`. 

### KITTI

Similarly, the training and testing of KITTI data set can be done by running
```bash
python train_KITTI.py

python test_KITTI.py --chosen_snapshot [exp_id] --descriptor FPFH --use_icp False
```

## Acknowledgments
We thank the authors of 
- [PointDSC](https://github.com/XuyangBai/PointDSC)
- [SC2PCR](https://github.com/ZhiChen902/SC2-PCR)
- [MAC](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)
- [VBReg](https://github.com/Jiang-HB/VBReg)
for open sourcing their methods.
