import numpy as np
import torch
from torch.utils.data import DataLoader

def collate_fn(list_data):
    min_num = 1e10
    # clip the pair having more correspondence during training.
    for ind, (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, scene, src_id, tgt_id) in enumerate(list_data):
        if len(gt_labels) < min_num:
            min_num = min(min_num, len(gt_labels))

    batched_corr_pos = []
    batched_src_keypts = []
    batched_tgt_keypts = []
    batched_gt_trans = []
    batched_gt_labels = []
    batched_src_normal = []
    batched_tgt_normal = []
    for ind, (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, scene, src_id, tgt_id) in enumerate(list_data):
        sel_ind = np.random.choice(len(gt_labels), min_num, replace=False)
        batched_corr_pos.append(corr_pos[sel_ind, :][None,:,:])
        batched_src_keypts.append(src_keypts[sel_ind, :][None,:,:])
        batched_tgt_keypts.append(tgt_keypts[sel_ind, :][None,:,:])
        batched_gt_trans.append(gt_trans[None,:,:])
        batched_gt_labels.append(gt_labels[sel_ind][None, :])
        batched_src_normal.append(src_normal[sel_ind, :][None, :, :])
        batched_tgt_normal.append(tgt_normal[sel_ind, :][None, :, :])
    
    batched_corr_pos = torch.from_numpy(np.concatenate(batched_corr_pos, axis=0))
    batched_src_keypts = torch.from_numpy(np.concatenate(batched_src_keypts, axis=0))
    batched_tgt_keypts = torch.from_numpy(np.concatenate(batched_tgt_keypts, axis=0))
    batched_gt_trans = torch.from_numpy(np.concatenate(batched_gt_trans, axis=0))
    batched_gt_labels = torch.from_numpy(np.concatenate(batched_gt_labels, axis=0))
    batched_src_normal = torch.from_numpy(np.concatenate(batched_src_normal, axis=0))
    batched_tgt_normal = torch.from_numpy(np.concatenate(batched_tgt_normal, axis=0))
    return batched_corr_pos, batched_src_keypts, batched_tgt_keypts, batched_src_normal, batched_tgt_normal, batched_gt_trans, batched_gt_labels, scene, src_id, tgt_id


def collate_fn1(list_data):
    min_num = 1e10
    # clip the pair having more correspondence during training.
    for ind, (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, filename) in enumerate(list_data):
        if len(gt_labels) < min_num:
            min_num = min(min_num, len(gt_labels))

    batched_corr_pos = []
    batched_src_keypts = []
    batched_tgt_keypts = []
    batched_gt_trans = []
    batched_gt_labels = []
    batched_src_normal = []
    batched_tgt_normal = []
    batched_filename = []
    for ind, (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, filename) in enumerate(list_data):
        sel_ind = np.random.choice(len(gt_labels), min_num, replace=False)
        batched_corr_pos.append(corr_pos[sel_ind, :][None, :, :])
        batched_src_keypts.append(src_keypts[sel_ind, :][None, :, :])
        batched_tgt_keypts.append(tgt_keypts[sel_ind, :][None, :, :])
        batched_gt_trans.append(gt_trans[None, :, :])
        batched_gt_labels.append(gt_labels[sel_ind][None, :])
        batched_src_normal.append(src_normal[sel_ind, :][None, :, :])
        batched_tgt_normal.append(tgt_normal[sel_ind, :][None, :, :])

    batched_corr_pos = torch.from_numpy(np.concatenate(batched_corr_pos, axis=0))
    batched_src_keypts = torch.from_numpy(np.concatenate(batched_src_keypts, axis=0))
    batched_tgt_keypts = torch.from_numpy(np.concatenate(batched_tgt_keypts, axis=0))
    batched_gt_trans = torch.from_numpy(np.concatenate(batched_gt_trans, axis=0))
    batched_gt_labels = torch.from_numpy(np.concatenate(batched_gt_labels, axis=0))
    batched_src_normal = torch.from_numpy(np.concatenate(batched_src_normal, axis=0))
    batched_tgt_normal = torch.from_numpy(np.concatenate(batched_tgt_normal, axis=0))
    return batched_corr_pos, batched_src_keypts, batched_tgt_keypts, batched_src_normal, batched_tgt_normal, batched_gt_trans, batched_gt_labels, filename


def collate_fn2(list_data):
    min_num = 1e10
    # clip the pair having more correspondence during training.
    for ind, (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels) in enumerate(list_data):
        if len(gt_labels) < min_num:
            min_num = min(min_num, len(gt_labels))

    batched_corr_pos = []
    batched_src_keypts = []
    batched_tgt_keypts = []
    batched_gt_trans = []
    batched_gt_labels = []
    batched_src_normal = []
    batched_tgt_normal = []
    for ind, (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels) in enumerate(list_data):
        sel_ind = np.random.choice(len(gt_labels), min_num, replace=False)
        batched_corr_pos.append(corr_pos[sel_ind, :][None,:,:])
        batched_src_keypts.append(src_keypts[sel_ind, :][None,:,:])
        batched_tgt_keypts.append(tgt_keypts[sel_ind, :][None,:,:])
        batched_gt_trans.append(gt_trans[None,:,:])
        batched_gt_labels.append(gt_labels[sel_ind][None, :])
        batched_src_normal.append(src_normal[sel_ind, :][None, :, :])
        batched_tgt_normal.append(tgt_normal[sel_ind, :][None, :, :])
    
    batched_corr_pos = torch.from_numpy(np.concatenate(batched_corr_pos, axis=0))
    batched_src_keypts = torch.from_numpy(np.concatenate(batched_src_keypts, axis=0))
    batched_tgt_keypts = torch.from_numpy(np.concatenate(batched_tgt_keypts, axis=0))
    batched_gt_trans = torch.from_numpy(np.concatenate(batched_gt_trans, axis=0))
    batched_gt_labels = torch.from_numpy(np.concatenate(batched_gt_labels, axis=0))
    batched_src_normal = torch.from_numpy(np.concatenate(batched_src_normal, axis=0))
    batched_tgt_normal = torch.from_numpy(np.concatenate(batched_tgt_normal, axis=0))
    return batched_corr_pos, batched_src_keypts, batched_tgt_keypts, batched_src_normal, batched_tgt_normal, batched_gt_trans, batched_gt_labels


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, fix_seed=True):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        multiprocessing_context=torch.multiprocessing.get_context("spawn")
    )

def get_dataloader_lc(dataset, batch_size, shuffle=True, num_workers=4, fix_seed=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn1,
        num_workers=num_workers,
        multiprocessing_context=torch.multiprocessing.get_context("spawn")
    )
    
def get_dataloader_train(dataset, batch_size, sampler=None, shuffle=True, num_workers=4, fix_seed=True):
    if sampler is not None:
        # Disable shuffling if a sampler is provided, as the sampler manages data shuffling
        shuffle = False
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn2,
        num_workers=num_workers,
        multiprocessing_context=torch.multiprocessing.get_context("spawn")
    )
