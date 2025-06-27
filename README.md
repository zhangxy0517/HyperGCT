# HyperGCT: A Dynamic Hyper-GNN-Learned Geometric Constraint for 3D Registration (ICCV 2025)
Source code of [HyperGCT](). 

## Introduction
Geometric constraints between feature matches are critical in 3D point cloud registration problems. Existing approaches typically model unordered matches as a consistency graph and sample consistent matches to generate hypotheses. However, explicit graph construction introduces noise, posing great challenges for handcrafted geometric constraints to render consistency. To overcome this, we propose HyperGCT, a flexible dynamic **Hyper**-**G**NN-learned geometric **C**onstrain**T** that leverages high-order consistency among 3D correspondences. To our knowledge, HyperGCT is the first method that mines robust geometric constraints from dynamic hypergraphs for 3D registration. By dynamically optimizing the hypergraph through vertex and edge feature aggregation, HyperGCT effectively captures the correlations among correspondences, leading to accurate hypothesis generation. Extensive experiments on 3DMatch, 3DLoMatch, KITTI-LC, and ETH show that HyperGCT achieves state-of-the-art performance. Furthermore, HyperGCT is robust to graph noise, demonstrating a significant advantage in terms of generalization. 

## Requirements

If you are using conda, you may configure PointDSC as:

    conda env create -f environment.yml
    conda activate pointdsc

CUDA 11.8

## Pretrained Model

We provide the pre-trained model of 3DMatch in `snapshot/PointDSC_3DMatch_release` and KITTI in `snapshot/PointDSC_KITTI_release`.


## Instructions to training and testing

### 3DMatch

The training and testing on 3DMatch dataset can be done by running
```bash
python train_3dmatch.py

python evaluation/test_3DMatch.py --chosen_snapshot [exp_id] --use_icp False
```
where the `exp_id` should be replaced by the snapshot folder name for testing (e.g. `PointDSC_3DMatch_release`).  The testing results will be saved in `logs/`. The training config can be changed in `config.py`. We also provide the scripts to test the traditional outlier rejection baselines on 3DMatch in `baseline_scripts/baseline_3DMatch.py`.

### KITTI

Similarly, the training and testing of KITTI data set can be done by running
```bash
python train_KITTI.py

python evaluation/test_KITTI.py --chosen_snapshot [exp_id] --use_icp False
```
We also provide the scripts to test the traditional outlier rejection baselines on KITTI in `baseline_scripts/baseline_KITTI.py`.


### Augmemented ICL-NUIM
The detailed guidance of evaluating our method in multiway registration tasks can be found in `multiway/README.md`

### 3DLoMatch
We also evaluate our method on a recently proposed benchmark 3DLoMatch following [OverlapPredator](https://github.com/ShengyuH/OverlapPredator),
```bash
python evaluation/test_3DLoMatch.py --chosen_snapshot [exp_id] --descriptor [fcgf/predator] --num_points 5000
```
If you want to evaluate `predator` descriptor with PointDSC, you first need to follow the offical instruction of [OverlapPredator](https://github.com/ShengyuH/OverlapPredator) to extract the features. 

## Acknowledgments
We thank the authors of 
- [PointDSC](https://github.com/XuyangBai/PointDSC)
- 
for open sourcing their methods.
