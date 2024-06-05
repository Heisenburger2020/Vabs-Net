import torch
import numpy as np
# import os
import pickle
import lmdb
import gzip
import numpy as np
import json
from tqdm import tqdm
# from pympler import asizeof
from multiprocessing import Pool
from collections import Counter
from scipy.spatial import distance_matrix
# pdb_assembly = json.load(open(os.path.join('/mnt/vepfs/users/lushuqi/protein/', "mmcif_assembly3_origin.json")) )
import gzip
from os.path import join
import lmdb
import os
import pickle
import torch
import numpy as np
from scipy.spatial import distance_matrix
from functools import lru_cache
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
from scipy.spatial import distance_matrix
from unicore.utils import batched_gather
import residue_constants as rc
from frame import Frame, Rotation

mode = 0
if mode:
    filename_original1 = r"/mnt/vepfs/fs_projects/uni-prot/protein_pocket/dataset/train/testAll4.lmdb"
else:
    filename_original1 = r"./store/dataset/train_merged_filtered.lmdb"
env_original1 = lmdb.open(
    filename_original1,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=True,
    meminit=False,
    max_readers=1,
    map_size=int(1000e9),
)
txn_original1 = env_original1.begin()
keys1 = list(txn_original1.cursor().iternext(values=False))
import warnings
# 关闭DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

def atom37_to_torsion_angles(
    aatype,
    all_atom_positions,
    all_atom_mask
):
    aatype = torch.from_numpy(aatype).long()
    all_atom_positions = torch.from_numpy(all_atom_positions)
    all_atom_mask = torch.from_numpy(all_atom_mask)
    
    if aatype.shape[-1] == 0:
        base_shape = aatype.shape
        torsion_angles_sin_cos = all_atom_positions.new_zeros(
            *base_shape, 7, 2
        )
        alt_torsion_angles_sin_cos = all_atom_positions.new_zeros(
            *base_shape, 7, 2
        )
        torsion_angles_mask = all_atom_positions.new_zeros(
            *base_shape, 7
        )
        return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask

    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(
        all_atom_mask[..., :2], dim=-1
    )
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(rc.chi_atom_indices, device=aatype.device)

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        num_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Frame.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic,
    )[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[..., None]
    )

    
    # consistent to uni-fold. use [1, 0] placeholder
    placeholder_torsions = torch.stack(
        [
            torch.ones(torsion_angles_sin_cos.shape[:-1]),
            torch.zeros(torsion_angles_sin_cos.shape[:-1]),
        ],
        dim=-1,
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[
        ..., None
    ] + placeholder_torsions * (1 - torsion_angles_mask[..., None])
    alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[
        ..., None
    ] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

    torsion_angles_sin_cos = np.array(torsion_angles_sin_cos)
    alt_torsion_angles_sin_cos = np.array(alt_torsion_angles_sin_cos)
    torsion_angles_mask = np.array(torsion_angles_mask)

    return torsion_angles_sin_cos, torsion_angles_mask

num_buckets = 10
num_tensors = 100
tensor_shape = (num_buckets, 7)  # 张量尺寸(50, 14)
statistics_tensor_all = np.zeros((num_buckets, tensor_shape[1]))
# 切分区间 [-1, 1] 成50份
buckets_limits = np.linspace(-np.pi, np.pi, num_buckets+1)
print(np.degrees(buckets_limits))
# for key in tqdm(keys1[:1000], total=len(keys1)):

def stat(key):
    statistics_tensor = np.zeros((num_buckets, tensor_shape[1]))
    data = txn_original1.get(key)
    try:
        data = gzip.decompress(data)
    except:
        pass
    data = pickle.loads(data)
    if mode:
        col_idx1, col_idx2 = 3, 4
        data["atom_masks"][:, [col_idx1, col_idx2]] = data["atom_masks"][:, [col_idx2, col_idx1]]
        data["atom_pos"][:, [col_idx1, col_idx2]] = data["atom_pos"][:, [col_idx2, col_idx1]]

    torsion, torsion_mask = atom37_to_torsion_angles(data["residue_type"], data["atom_pos"], data["atom_masks"])
    torsion *= torsion_mask.reshape(-1, 7, 1)
    torsion += (1 - torsion_mask.reshape(-1, 7, 1)) * 500
    sin_values = torsion[..., 0]
    cos_values = torsion[..., 1]
    # 使用arctan2函数计算角度（弧度值）
    radian_values = np.arctan2(sin_values, cos_values)
    for j in range(tensor_shape[1]):
        # 对于每一列，我们计算属于哪个桶
        for i in range(num_buckets):
            # 根据边界来统计每个区间内的数值出现的次数
            mask = (radian_values[:, j] >= buckets_limits[i]) & (radian_values[:, j] < buckets_limits[i+1])
            statistics_tensor[i, j] += mask.sum().item()
    return statistics_tensor
if not mode:
    keys1 = keys1[::100]
with Pool(56) as pool:
    for ret in tqdm(pool.imap(stat, keys1), total=len(keys1)):
        statistics_tensor_all += ret

x = torch.from_numpy(statistics_tensor_all)
sum_x = torch.sum(x, dim=0, keepdim=True)  # 求和
normalized_x = torch.div(x, sum_x)  # 除以和
torch.set_printoptions(precision=4)
print(normalized_x)

# [4.241023e-01, 5.339414e-02, 5.850743e-02, 1.430713e-01, 9.766694e-02, 3.644090e-02, 2.702102e-02],
# [2.794911e-04, 1.945780e-01, 2.256833e-02, 1.540983e-02, 1.122382e-02, 6.574225e-03, 4.497151e-03],
# [2.921952e-04, 2.624164e-01, 3.253447e-02, 1.053090e-01, 5.592855e-02, 1.795049e-02, 1.328819e-02],
# [2.985473e-04, 4.168376e-01, 2.248830e-01, 2.975380e-01, 7.749963e-02, 3.662510e-02, 8.930783e-03],
# [2.248633e-03, 2.763080e-03, 1.572097e-01, 2.563647e-02, 4.955759e-02, 2.148851e-02, 3.938183e-04],
# [4.084381e-03, 3.576124e-03, 1.969727e-02, 2.402307e-02, 2.989208e-02, 1.997040e-02, 3.747626e-04],
# [8.581646e-03, 3.360795e-02, 6.015257e-03, 2.503811e-01, 4.347945e-01, 8.010963e-01, 9.013612e-01],
# [3.080754e-03, 2.279700e-02, 5.475345e-03, 2.296229e-02, 6.637744e-02, 1.740423e-02, 1.112854e-02],
# [2.667869e-04, 4.535263e-03, 2.928484e-01, 2.521724e-03, 1.552406e-02, 6.783838e-03, 5.214917e-03],
# [5.567653e-01, 5.494401e-03, 1.802608e-01, 1.131473e-01, 1.615354e-01, 3.566597e-02, 2.778960e-02]

# [2.9773e-01, 3.1164e-02, 1.2540e-02, 1.5213e-01, 1.0770e-01, 6.1553e-02, 3.8458e-02],
# [2.2275e-04, 1.4011e-01, 3.3813e-03, 7.9056e-03, 8.7731e-03, 4.3277e-03, 3.8580e-03],
# [7.0233e-05, 2.5182e-01, 1.5332e-03, 1.0837e-01, 6.1751e-02, 8.3744e-03, 6.8919e-03],
# [1.3995e-04, 4.5670e-01, 3.1814e-01, 3.0614e-01, 6.3565e-02, 1.9379e-02, 2.0355e-03],
# [1.5531e-03, 1.9169e-03, 1.6713e-01, 2.1491e-02, 6.6161e-02, 1.6128e-02, 6.4896e-05],
# [1.6694e-03, 1.2709e-03, 5.1408e-02, 2.4442e-02, 3.0991e-02, 1.2815e-02, 6.6273e-05],
# [6.3324e-02, 8.8421e-02, 7.6565e-02, 2.8562e-01, 4.4574e-01, 7.9400e-01, 8.9752e-01],
# [1.0793e-03, 2.0839e-02, 4.0148e-02, 1.5438e-02, 2.6664e-02, 9.1197e-03, 5.5835e-03],
# [5.0303e-03, 3.2192e-03, 2.1849e-01, 6.2572e-04, 6.5640e-03, 7.4832e-03, 4.1501e-03],
# [6.2918e-01, 4.5358e-03, 1.1066e-01, 7.7853e-02, 1.8209e-01, 6.6824e-02, 4.1376e-02],