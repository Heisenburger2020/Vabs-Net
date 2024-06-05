# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import os
import pickle
import torch
import numpy as np
from scipy.spatial import distance_matrix
from functools import lru_cache
from torch.utils.data.dataloader import default_collate
from unicore.data import BaseWrapperDataset, UnicoreDataset
from . import data_utils
from unicore.distributed.utils import get_data_parallel_rank
import torch.nn as nn
from scipy.spatial.transform import Rotation
import gzip
import torch.nn.functional as F
from unicore.utils import batched_gather
from ..models.frame import Frame, Rotation
from . import residue_constants as rc


atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {i:atom_type for i, atom_type in enumerate(atom_types)}
atom2num = {atom_type:i for i, atom_type in enumerate(atom_types)}


atom_edge_encode_dict = {}
cnt = 0
for i in range(38):
    for j in range(i, 38):
        atom_edge_encode_dict[(i, j)] = cnt
        atom_edge_encode_dict[(j, i)] = cnt
        cnt += 1
res_edge_encode_dict = {}
cnt = 0
for i in range(23):
    for j in range(i, 23):
        res_edge_encode_dict[(i, j)] = cnt
        res_edge_encode_dict[(j, i)] = cnt
        cnt += 1

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def no_pos_dist_array(n):
    arr = np.arange(n)
    return np.minimum(arr, n - arr - 1)


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

def remove_center(
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    eps: float = 1e-8,
    ca_idx: int=1
):
    pos = all_atom_positions[..., ca_idx, :]      # use ca only.
    msk = all_atom_mask[..., ca_idx, None]
    centre = (pos * msk).sum(dim=-2, keepdim=True) / (msk.sum(dim=-2, keepdim=True)).clamp_min(eps)
    all_atom_positions -= centre[..., :, None, :]
    all_atom_positions *= all_atom_mask[..., None]

    return all_atom_positions

def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def compute_sin_cos(vectors):
    angle_xy = torch.atan2(torch.norm(vectors[:, 2:3], dim=1), torch.norm(vectors[:, :2], dim=1))
    sin_xy = torch.sin(angle_xy)
    cos_xy = torch.cos(angle_xy)
    
    angle_xz = torch.atan2(torch.norm(vectors[:, 1:2], dim=1), torch.norm(vectors[:, [0, 2]], dim=1))
    sin_xz = torch.sin(angle_xz)
    cos_xz = torch.cos(angle_xz)
    
    angle_yz = torch.atan2(torch.norm(vectors[:, :2], dim=1), torch.norm(vectors[:, [1, 2]], dim=1))
    sin_yz = torch.sin(angle_yz)
    cos_yz = torch.cos(angle_yz)
    
    result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)
    return result

def compute_sin_cos_res(knn,start,dest,X,use_local_coord):
    vectors = dest-start
    if use_local_coord:
        # use true atoms only
        num_real = X.shape[0]
        num = vectors.shape[0]
        real_vec = vectors[:-2 * num_real,:]
        assert real_vec.shape[0] == num-2*num_real

        X = X.unsqueeze(0)
        V = X.clone()
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
        dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
        U = _normalize(dX, dim=-1)
        u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
        n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
        b_1 = _normalize(u_0 - u_1, dim=-1)
        
        n_0 = n_0[:,::3,:]
        b_1 = b_1[:,::3,:]
        if b_1.shape[1] >= real_vec.shape[0]//knn:
             err = b_1.shape[1] - real_vec.shape[0]//knn+1
             n_0 = n_0[:,:-err,:]
             b_1 = b_1[:,:-err,:]
        X = X[:,::3,:] #[B,N,3]
        Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
        Q = Q.view(list(Q.shape[:2]) + [9]) # [B, N, 3,3]
        Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [B, N, 9]
        Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2) # [B, N, 1, 3, 3]
        
        real_vec = real_vec.reshape(-1,knn,3).unsqueeze(0).unsqueeze(3) # [B,N,30,1,3]
        dU = torch.matmul(Q[:,:,:,None,:,:], real_vec[...,None]).squeeze(-1) # [B, N, 30,1,3] 邻居的相对坐标
        B, N, K = dU.shape[:3]
        E_direct = _normalize(dU, dim=-1)
        E_direct = E_direct.reshape(N*K,-1)
        vectors = torch.cat((E_direct,vectors[num - 2 * num_real:,:]),dim=0)
       
    angle_xy = torch.atan2(torch.norm(vectors[:, 2:3], dim=1), torch.norm(vectors[:, :2], dim=1))
    sin_xy = torch.sin(angle_xy)
    cos_xy = torch.cos(angle_xy)
    
    angle_xz = torch.atan2(torch.norm(vectors[:, 1:2], dim=1), torch.norm(vectors[:, [0, 2]], dim=1))
    sin_xz = torch.sin(angle_xz)
    cos_xz = torch.cos(angle_xz)
    
    angle_yz = torch.atan2(torch.norm(vectors[:, :2], dim=1), torch.norm(vectors[:, [1, 2]], dim=1))
    sin_yz = torch.sin(angle_yz)
    cos_yz = torch.cos(angle_yz)
    
    result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)
    return result

def compute_sin_cos_aa(knn,start,dest,X,use_local_coord):
    vectors = dest-start
    if use_local_coord:
        num_real = X.shape[0]
        num = vectors.shape[0]
        step = num_real * 32
        num_remove = num_real * 2

        # use true atoms only
        num_segments = vectors.shape[0] // step
        segments = np.split(vectors, num_segments)
        trimmed_segments = [segment[:-num_remove, :] for segment in segments]
        real_vec = torch.tensor(np.concatenate(trimmed_segments, axis=0))

        X = X.unsqueeze(0)
        V = X.clone()
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
        dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
        U = _normalize(dX, dim=-1)
        u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
        n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
        b_1 = _normalize(u_0 - u_1, dim=-1)
        
        n_0 = n_0[:,::3,:]
        b_1 = b_1[:,::3,:]
        if b_1.shape[1] >= real_vec.shape[0]//knn:
            err = b_1.shape[1] - real_vec.shape[0]//knn+1
            n_0 = n_0[:,:-err,:]
            b_1 = b_1[:,:-err,:]
        X = X[:,::3,:] #[B,N,3]
        Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
        Q = Q.view(list(Q.shape[:2]) + [9]) # [B, N, 3,3]
        Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [B, N, 9]
        Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2) # [B, N, 1, 3, 3]

        real_vec = real_vec.reshape(-1,knn,4,3).unsqueeze(0)  # [B,N,30,4,3]
        dU = torch.matmul(Q[:,:,:,None,:,:], real_vec[...,None]).squeeze(-1) # [B,N,30,4,3] 邻居的相对坐标
        B, N, K = dU.shape[:3]
        E_direct = _normalize(dU, dim=-1)
        E_direct = E_direct.reshape(N*K*4,-1)
        vir_segments = [segment[num_real * knn:, :] for segment in segments]
        real_segments = np.split(E_direct, num_segments)
        trimmed_segments = [torch.cat((a,b),dim=0) for a,b in zip(vir_segments,real_segments)]
        vectors = torch.tensor(np.concatenate(trimmed_segments, axis=0))
    
    angle_xy = torch.atan2(torch.norm(vectors[:, 2:3], dim=1), torch.norm(vectors[:, :2], dim=1))
    sin_xy = torch.sin(angle_xy)
    cos_xy = torch.cos(angle_xy)
    
    angle_xz = torch.atan2(torch.norm(vectors[:, 1:2], dim=1), torch.norm(vectors[:, [0, 2]], dim=1))
    sin_xz = torch.sin(angle_xz)
    cos_xz = torch.cos(angle_xz)
    
    angle_yz = torch.atan2(torch.norm(vectors[:, :2], dim=1), torch.norm(vectors[:, [1, 2]], dim=1))
    sin_yz = torch.sin(angle_yz)
    cos_yz = torch.cos(angle_yz)
    
    result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)
    return result

def make_backbone_frames(
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    eps: float = 1e-8,
):
    num_batch_dim = len(all_atom_positions.shape[:-2])

    gt_frames = Frame.from_3_points(
        p_neg_x_axis=all_atom_positions[..., 2, :],
        origin=all_atom_positions[..., 1, :],
        p_xy_plane=all_atom_positions[..., 0, :],
        eps=eps,
    )

    rots = torch.diag(
        torch.tensor(
            (-1., 1., -1.),
            dtype=all_atom_positions.dtype,
            device=all_atom_positions.device
        )
    )
    rots = torch.tile(rots, (1,) * (num_batch_dim + 2))
    gt_frames = gt_frames.compose(Frame(Rotation(rots), None))

    gt_exists = torch.min(all_atom_mask[..., :3], dim=-1, keepdim=False)[0]

    return gt_frames.to_tensor_4x4(), gt_exists


class LMDB2Dataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

        self.env = self.connect_db(self.db_path)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        return data

class ClusteredDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

        self.env = self.connect_db(self.db_path)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(gzip.decompress(datapoint_pickled))
        ret = data[np.random.randint(len(data))]
        return ret 
    
class ClusteredMergeDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        # assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

        lmdbs_path = [os.path.join(self.db_path, p) for p in os.listdir(self.db_path)]
        self.envs = self.connect_db(lmdbs_path)
        self._keys = []
        self._env_idx = []
        for i, env in enumerate(self.envs):
            with env.begin() as txn:
                keys = list(txn.cursor().iternext(values=False))
                self._keys += keys
                self._env_idx += [i for _ in range(len(keys))]

    def connect_db(self, lmdb_paths=None):
        envs = [
            lmdb.open(
                lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=128,
            ) for lmdb_path in lmdb_paths]
        return envs

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):

        datapoint_pickled = self.envs[self._env_idx[idx]].begin().get(self._keys[idx])
        data = pickle.loads(gzip.decompress(datapoint_pickled))
        ret = data[np.random.randint(len(data))]
        return ret 

def remove_elements_by_index(A, B):
    return [element for index, element in enumerate(A) if index not in B]

class ZipLMDB2Dataset:
    def __init__(self, db_path, reverse=False, remove_list=[]):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

        self.env = self.connect_db(self.db_path)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        if len(remove_list) > 0:
            self._keys = remove_elements_by_index(self._keys, remove_list)
        if reverse:
            self._keys.reverse()

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=128,
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(gzip.decompress(datapoint_pickled))
        return data

def print_all_shape(*tensors):
    print("shape>>", [_.shape for _ in tensors])

class IFPreDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(self, dataset, args, split, crop_rational=1/4, is_train=False):
        super().__init__(dataset)
        self.dataset = dataset
        self.is_train = is_train
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.set_epoch(None)
        self.keep = args.keep
        self.crop = args.crop
        # rotate sample
        # current solution: randomly sample rotation
        self.n_sample_random = 4
        self.knn = args.knn
        self.num_rbf = 16
        self.split = split

        self.funcs = [torch.sin, torch.cos]
        self.xyz_N_freqs = 10
        self.xyz_freq_bands = 2**torch.linspace(0, self.xyz_N_freqs-1, self.xyz_N_freqs)
        self.theta_N_freqs = 4
        self.theta_freq_bands = 2**torch.linspace(0, self.theta_N_freqs-1, self.theta_N_freqs)


    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def GetRotationMatrix(self, theta, phi, gamma):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)], 
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)], 
                       [0, 1, 0],  
                       [-np.sin(phi), 0, np.cos(phi)]])        
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])       
        return np.dot(Rz, np.dot(Ry, Rx))

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    def sample_trans(self, before, rotate_op, shift_op, scale_op):
        ss = before * scale_op.reshape(1, 3) + shift_op.reshape(-1, 3)
        after = rotate_op.T @ ss.transpose(1, 0)
        return after

    def GetRotationMatrix(self, theta, phi, gamma):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)], 
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)], 
                       [0, 1, 0],  
                       [-np.sin(phi), 0, np.cos(phi)]])        
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])       
        return np.dot(Rz, np.dot(Ry, Rx))
    
    def get_nerf_feat(self, x, freq_bands):
        out = [x]
        for freq in freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.stack(out, -1)

    def get_random_fourier_feat(self, coords):
        assert self.rand_freqs is not None
        # k = coords . rand_freqs
        # expand rand_freqs with singleton dimension along the batch dimensions
        # e.g. dim (1, ..., 1, n_rand_feats, 3)
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2

        kxkykz = coords[..., None, 0:3] * freqs  # compute the x,y,z components of k
        k = kxkykz.sum(-1)  # compute k
        s = torch.sin(k)
        c = torch.cos(k)
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        print(x.shape, coords.shape)
        return x

    def compute_sin_cos(self, vectors):
        angle_xy = torch.atan2(torch.norm(vectors[:, 2:3], dim=1), torch.norm(vectors[:, :2], dim=1))
        sin_xy = torch.sin(angle_xy)
        cos_xy = torch.cos(angle_xy)
        
        angle_xz = torch.atan2(torch.norm(vectors[:, 1:2], dim=1), torch.norm(vectors[:, [0, 2]], dim=1))
        sin_xz = torch.sin(angle_xz)
        cos_xz = torch.cos(angle_xz)
        
        angle_yz = torch.atan2(torch.norm(vectors[:, :2], dim=1), torch.norm(vectors[:, [1, 2]], dim=1))
        sin_yz = torch.sin(angle_yz)
        cos_yz = torch.cos(angle_yz)
        
        if not self.args.use_nerf_encoding:
            result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)
        elif not self.args.random_fourier:
            result = []
            result.append(self.get_nerf_feat(torch.atan2(sin_xy, cos_xy), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_xz, cos_xz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_yz, cos_yz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(vectors, self.xyz_freq_bands).reshape(-1, 63))
            result = torch.cat(result, dim=-1)
        else:
            assert 0
            result = []
            result.append(self.get_random_fourier_feat(torch.atan2(sin_xy, cos_xy), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_xz, cos_xz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_yz, cos_yz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.norm(vectors, dim=-1, keepdim=True), self.xyz_freq_bands).reshape(-1, 63))
            result = torch.cat(result, dim=-1)
        return result

    def compute_sin_cos_res(self, knn,start,dest,X,use_local_coord, num_real_edges):
        vectors = dest-start
        if use_local_coord:
            # use true atoms only
            if self.args.use_virtual_node and not self.args.num_virtual_point:
                num_real = X.shape[0]
                num = vectors.shape[0]
                real_vec = vectors[:-2 * num_real,:]
                assert real_vec.shape[0] == num-2*num_real
            elif self.args.num_virtual_point:
                num_real = X.shape[0]
                real_vec = vectors[:num_real_edges,:]
                # pass
            else:
                real_vec = vectors

            X = X.unsqueeze(0)
            V = X.clone()
            X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
            dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
            U = _normalize(dX, dim=-1)
            u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
            n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
            b_1 = _normalize(u_0 - u_1, dim=-1)

            n_0 = n_0[:,::3,:]
            b_1 = b_1[:,::3,:]
            if b_1.shape[1] >= real_vec.shape[0]//knn:
                assert 0
                err = b_1.shape[1] - real_vec.shape[0]//knn+1
                n_0 = n_0[:,:-err,:]
                b_1 = b_1[:,:-err,:]
            X = X[:,::3,:] #[B,N,3]
            Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
            Q = Q.view(list(Q.shape[:2]) + [9]) # [B, N, 3,3]
            Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [B, N, 9]
            Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2) # [B, N, 1, 3, 3]

            real_vec = real_vec.reshape(-1,knn,3).unsqueeze(0).unsqueeze(3) # [B,N,30,1,3]
            dU = torch.matmul(Q[:,:,:,None,:,:], real_vec[...,None]).squeeze(-1) # [B, N, 30,1,3] 邻居的相对坐标
            B, N, K = dU.shape[:3]
            E_direct = _normalize(dU, dim=-1)
            E_direct = E_direct.reshape(N*K,-1)
            if self.args.use_virtual_node:
                vectors = torch.cat((E_direct,vectors[num_real_edges:,:]),dim=0)
            else:
                vectors = E_direct

        angle_xy = torch.atan2(torch.norm(vectors[:, 2:3], dim=1), torch.norm(vectors[:, :2], dim=1))
        sin_xy = torch.sin(angle_xy)
        cos_xy = torch.cos(angle_xy)

        angle_xz = torch.atan2(torch.norm(vectors[:, 1:2], dim=1), torch.norm(vectors[:, [0, 2]], dim=1))
        sin_xz = torch.sin(angle_xz)
        cos_xz = torch.cos(angle_xz)

        angle_yz = torch.atan2(torch.norm(vectors[:, :2], dim=1), torch.norm(vectors[:, [1, 2]], dim=1))
        sin_yz = torch.sin(angle_yz)
        cos_yz = torch.cos(angle_yz)

        if not self.args.use_nerf_encoding:
            result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)
        else:
            result = []
            result.append(self.get_nerf_feat(torch.atan2(sin_xy, cos_xy), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_xz, cos_xz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_yz, cos_yz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(vectors, self.xyz_freq_bands).reshape(-1, 63))
            result = torch.cat(result, dim=-1)
        return result


    def compute_sin_cos_aa(self, knn,start,rows_aa,dest,cols_aa,residue_pos_all,X,use_local_coord):
        vectors = dest-start
        if use_local_coord:
            num_real = X.shape[0]
            num = vectors.shape[0]
            step = num_real * 32
            # num_remove = num_real * 2
            if self.args.use_virtual_node:
                X = torch.cat([X, torch.zeros((1, 4, 3))], axis=0)
            # # use true atoms only
            # num_segments = vectors.shape[0] // step
            # segments = np.split(vectors, num_segments)
            # trimmed_segments = [segment[:-num_remove, :] for segment in segments]
            # real_vec = torch.tensor(np.concatenate(trimmed_segments, axis=0))

            X = X.unsqueeze(0)
            V = X.clone()
            X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
            dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
            U = _normalize(dX, dim=-1)
            u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
            n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
            b_1 = _normalize(u_0 - u_1, dim=-1)

            n_0 = n_0[:,::3,:]
            b_1 = b_1[:,::3,:]
            # if b_1.shape[1] >= real_vec.shape[0]//knn:
            #     err = b_1.shape[1] - real_vec.shape[0]//knn+1
            #     n_0 = n_0[:,:-err,:]
            #     b_1 = b_1[:,:-err,:]
            X = X[:,::3,:] #[B,N,3]
            Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
            Q = Q.view(list(Q.shape[:2]) + [9]) # [B, N, 3,3]
            Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [B, N, 9]
            Q = Q.view(list(Q.shape[:2]) + [3,3]) # [B, N, 3, 3]

            dU = torch.matmul(Q[:,residue_pos_all[rows_aa],:,:], vectors[...,None]).squeeze(-1) # [B,N,30,4,3] 邻居的相对坐标
            # B, N, K = dU.shape[:3]
            E_direct = _normalize(dU, dim=-1)
            vectors = E_direct.squeeze(0)
            # E_direct = E_direct.reshape(N*K*4,-1)
            # vir_segments = [segment[num_real * knn:, :] for segment in segments]
            # real_segments = np.split(E_direct, num_segments)
            # trimmed_segments = [torch.cat((a,b),dim=0) for a,b in zip(vir_segments,real_segments)]
            # vectors = torch.tensor(np.concatenate(trimmed_segments, axis=0))

        angle_xy = torch.atan2(torch.norm(vectors[:, 2:3], dim=1), torch.norm(vectors[:, :2], dim=1))
        sin_xy = torch.sin(angle_xy)
        cos_xy = torch.cos(angle_xy)

        angle_xz = torch.atan2(torch.norm(vectors[:, 1:2], dim=1), torch.norm(vectors[:, [0, 2]], dim=1))
        sin_xz = torch.sin(angle_xz)
        cos_xz = torch.cos(angle_xz)

        angle_yz = torch.atan2(torch.norm(vectors[:, :2], dim=1), torch.norm(vectors[:, [1, 2]], dim=1))
        sin_yz = torch.sin(angle_yz)
        cos_yz = torch.cos(angle_yz)

        result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)

        if not self.args.use_nerf_encoding:
            result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)
        else:
            result = []
            result.append(self.get_nerf_feat(torch.atan2(sin_xy, cos_xy), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_xz, cos_xz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_yz, cos_yz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(vectors, self.xyz_freq_bands).reshape(-1, 63))
            result = torch.cat(result, dim=-1)
        return result

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):
            data = self.dataset[idx]
            try:
                data = gzip.decompress(data)
            except:
                pass
            try:
                data = pickle.loads(data)
            except:
                pass
            '''
            # atom_pred_pos_index
            # atom_pos
            # atom_type
            # residue_type
            # edge_weight
            # edge_res_attr
            # edge_atom_attr
            # edge_index
            # edge_diff
            # atom_pos_origin
            # residue_type_origin
            # atom_type_origin
            '''
            is_nan = np.isnan(np.sum(data["atom_pos"], axis=(1, 2)))
            mask_nan = ~is_nan
            data["atom_pos"] = data["atom_pos"][mask_nan]
            data["atom_masks"] = data["atom_masks"][mask_nan]
            data["residue_type"] = data["residue_type"][mask_nan]
            eps = 1e-12
            rdm = np.random.uniform(0, 1, size=(1))
            rdm_residue = np.random.uniform(0, 1, size=(1))
            residue_only_mode = False

            mode_3d = True
            mode_esm = True
            if rdm <= 0.5:
                mode_3d = True
                mode_esm = True
            elif rdm < 0.75:
                mode_3d = False
            else:
                mode_esm = False

            try:
                data_flag = data["dataset"]
                data_flag = 0
                # data["atom_pos"] = data["atom_pos"] + np.random.randn(*data["atom_pos"].shape) * 0.1
            except:
                data_flag = 0

            try:
                torsion, torsion_mask = atom37_to_torsion_angles(data["residue_type"], data["atom_pos"], data["atom_masks"])
            except:
                torsion = np.zeros((data["residue_type"].shape[0], 7, 2))
                torsion_mask = np.zeros((data["residue_type"].shape[0], 7))

            num_res = data["residue_type"].shape[0]
            residue_type_all = torch.from_numpy(data["residue_type"]).unsqueeze(-1).expand(-1, 37).long()
            atom_masks_all = torch.from_numpy(data["atom_masks"]).bool()
            atom_pos_all = torch.from_numpy(data["atom_pos"])
            atom_type_all = torch.from_numpy(np.arange(37)).unsqueeze(0).expand(num_res, -1)

            def legal_check(atom_pos_all, atom_masks_all, CC_DIST=1.2, CO_DIST=1):
                legal_flag = torch.ones(atom_masks_all.shape[0])
                residue_index = torch.arange(atom_masks_all.shape[0]).unsqueeze(1).expand(-1, 37).reshape(-1)[atom_masks_all.reshape(-1)]
                atom_pos_ = atom_pos_all.reshape(-1, 3)[atom_masks_all.reshape(-1)]
                illegal = torch.sum(torch.abs(atom_pos_) < 0.001, dim=-1).long() == 3
                illgeal_index = torch.unique(residue_index[illegal])
                legal_flag[illgeal_index] = 0

                dist = distance_matrix(atom_pos_, atom_pos_)
                dist = torch.from_numpy(dist) + torch.eye(atom_pos_.shape[0]) * 2
                dist = dist < 1
                illegal = torch.any(dist, dim=-1)
                illgeal_index = torch.unique(residue_index[illegal])
                legal_flag[illgeal_index] = 0
                
                ret_flag = True

                if legal_flag.sum() / legal_flag.shape[0] < 0.7:
                    print(f"too much illegal residue: {idx} {legal_flag.sum() / legal_flag.shape[0]}, {legal_flag.sum()}, {legal_flag.sum().shape}")
                    ret_flag = False
                    legal_flag = torch.ones(atom_masks_all.shape[0])
                return legal_flag.bool(), ret_flag

            legal_flag, flag_pred = legal_check(atom_pos_all, atom_masks_all)

            residue_type_all = residue_type_all[legal_flag]
            atom_masks_all = atom_masks_all[legal_flag]
            atom_type_all = atom_type_all[legal_flag]
            torsion = torsion[legal_flag]
            torsion_mask = torsion_mask[legal_flag]

            try:
                esm_feat = data["esm_feat"].half()
                esm_cls_feat = data["esm_cls_feat"].half()
            except:
                esm_feat = torch.zeros(atom_pos_all.shape[0], self.args.esm_dim).half()
                esm_cls_feat = torch.zeros(self.args.esm_dim).half()
            esm_feat = esm_feat[legal_flag]
            atom_pos_all = atom_pos_all[legal_flag]

            # mask if no alpha C
            C_mask = torch.zeros(atom_masks_all.shape).bool()
            C_mask[:, 1] = True
            C_mask = atom_masks_all.long() & C_mask
            atom_pos_all = atom_pos_all[C_mask[:, 1].bool(), :, :]
            atom_type_all = atom_type_all[C_mask[:, 1].bool(), :]
            residue_type_all = residue_type_all[C_mask[:, 1].bool(), :]
            atom_masks_all = atom_masks_all[C_mask[:, 1].bool(), :]
            esm_feat = esm_feat[C_mask[:, 1].bool(), :]
            torsion, torsion_mask = torsion[C_mask[:, 1].bool(), :, :], torsion_mask[C_mask[:, 1].bool(), :]
            # crop            
            length_threshold = self.args.cutoff
            if atom_pos_all.shape[0] > length_threshold:
                residue_type_all = residue_type_all[:length_threshold]
                atom_type_all = atom_type_all[:length_threshold]
                atom_pos_all = atom_pos_all[:length_threshold, :]
                atom_masks_all = atom_masks_all[:length_threshold, :]
                esm_feat = esm_feat[:length_threshold, :]
                torsion, torsion_mask = torsion[:length_threshold, :], torsion_mask[:length_threshold, :]

            residue_pos = torch.from_numpy(np.arange(atom_pos_all.shape[0]))
            residue_pos_all = torch.from_numpy(np.arange(atom_pos_all.shape[0])).unsqueeze(1).expand(-1, 37)

            theta, phi, gamma = np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)
            rot = self.GetRotationMatrix(theta, phi, gamma)
            trans = np.random.uniform(-20, 20, size=3)
            atom_pos_all = torch.from_numpy(np.dot(atom_pos_all.reshape(-1, 3), rot) + trans).reshape(-1, 37, 3)

            atom_pos_all = remove_center(atom_pos_all, atom_masks_all, eps)

            # save for label
            atom_pos_all_origin = atom_pos_all.clone().reshape(-1, 3)
            residue_type_origin = residue_type_all.clone().reshape(-1)

            # add noise
            n_atom = atom_pos_all.shape[0]
            atom_type_origin = atom_type_all.clone()
            
            # only save alpahc 
            if self.args.sas_pred:
                import freesasa
                radius_dic = {
                    'N': 1.6,
                    'C': 1.7,
                    'O': 1.4,
                    'S': 1.8,
                }
                atom_radius = torch.Tensor([radius_dic[a[0]] for a in atom_types])
                atoms_ = atom_pos_all.reshape(-1, 3)
                atoms_m_ = atom_masks_all.reshape(-1).bool()
                atoms_r_ = np.tile(atom_radius.reshape(1, 37), (atom_pos_all.shape[0], 1)).reshape(-1)
                sas = torch.zeros(atoms_.shape[0])
                atoms_id_ = torch.arange(atoms_.shape[0])
                atoms_ = atoms_[atoms_m_].reshape(-1)
                atoms_id_ = atoms_id_[atoms_m_]
                atoms_r_ = atoms_r_[atoms_m_]
                sas_atom = freesasa.calcCoord(atoms_, atoms_r_)
                for id_s in range(len(atoms_id_)):
                    sas[atoms_id_[id_s]] = sas_atom.atomArea(id_s)
                sas = sas.reshape(-1, 37)

            def create_span_mask(sequence_length, span_lambda=6, mask_ratio=0.3):
                num_to_mask = int(sequence_length * mask_ratio)
                sum_masked = 0
                idx_current = 0
                span_mask_indices = []
                while sum_masked < num_to_mask:
                    num_mask_left = num_to_mask - sum_masked
                    span_length = torch.poisson(torch.tensor([float(span_lambda)])).int().item()
                    span_length = min(span_length, num_mask_left) # 确保不超过比例
                    assert idx_current <= sequence_length - num_mask_left - span_length, (idx_current, sequence_length - num_mask_left - span_length)
                    start_idx = torch.randint(0, sequence_length - span_length, (1,)).item()
                    span_mask_indices.extend(range(start_idx, start_idx + span_length))
                    sum_masked += span_length
                return torch.tensor(span_mask_indices, dtype=torch.long)

            # res_type_mask_prob
            # res_noise_scale
            # select_num = int(self.args.res_type_mask_prob * n_atom)
            # if not self.args.span_mask:
            #     index_select = np.random.choice(np.arange(n_atom), select_num, replace=False)
            # else:
            #     index_select = np.array(set(list(create_span_mask(n_atom))))
            res_type_list = list(set(list(create_span_mask(n_atom))))
            index_select_type = np.array(res_type_list)
            residue_type_all[index_select_type] = torch.ones(len(index_select_type), 37, dtype=torch.long) * 22
            esm_feat[index_select_type] = torch.zeros(len(index_select_type), esm_feat.shape[1], dtype=torch.long).type(torch.half)
            
            side_chain_mask = torch.zeros(len(index_select_type), 37, dtype=torch.long)
            side_chain_mask[:, [1]] = 1
            # side_chain_mask[:, [0, 1, 2, 4]] = 1
            atom_masks_all[index_select_type] = (atom_masks_all[index_select_type] * side_chain_mask).bool()

            # select_num = int(self.args.res_pos_mask_prob * n_atom)
            # if not self.args.mask_same:
            #     if not self.args.span_mask:
            #         index_select = np.random.choice(np.arange(n_atom), select_num, replace=False)
            #     else:
            #         index_select = np.array(set(list(create_span_mask(n_atom))))

            # span_mask_mask = create_span_mask(len(res_type_list), span_lambda=6, mask_ratio=0.5)

            # index_select = np.array(list(set(list(create_span_mask(n_atom)) + list(torch.tensor(res_type_list)[span_mask_mask]))))
            index_select = index_select_type
            if flag_pred:
                atom_pos_all[index_select] += torch.from_numpy(np.random.randn(len(index_select), 37, 3)).float() * self.args.res_noise_scale #[N_RES, 37, 3]
            # atom_pos_all[index_select_type] = atom_pos_all[index_select_type, 1, :].unsqueeze(1).repeat(1, 37, 1)
            atom_pos_pred_index = np.zeros((n_atom, 37))
            if flag_pred:
                atom_pos_pred_index[index_select] += 1
            atom_pos_pred_index = torch.from_numpy(atom_pos_pred_index).long()

            atom_pos_all37 = atom_pos_all.clone()
            atom_masks_all37 = atom_masks_all.clone()
            # only save alpahc 
            if self.args.use_sas and not self.args.sas_pred:
                import freesasa
                radius_dic = {
                    'N': 1.6,
                    'C': 1.7,
                    'O': 1.4,
                    'S': 1.8,
                }
                atom_radius = torch.Tensor([radius_dic[a[0]] for a in atom_types])
                atoms_ = atom_pos_all.reshape(-1, 3)
                atoms_m_ = atom_masks_all.reshape(-1).bool()
                atoms_r_ = np.tile(atom_radius.reshape(1, 37), (atom_pos_all.shape[0], 1)).reshape(-1)
                sas = torch.zeros(atoms_.shape[0])
                atoms_id_ = torch.arange(atoms_.shape[0])
                atoms_ = atoms_[atoms_m_].reshape(-1)
                atoms_id_ = atoms_id_[atoms_m_]
                atoms_r_ = atoms_r_[atoms_m_]
                sas_atom = freesasa.calcCoord(atoms_, atoms_r_)
                for id_s in range(len(atoms_id_)):
                    sas[atoms_id_[id_s]] = sas_atom.atomArea(id_s)
                sas = sas.reshape(-1, 37)
        
            atom_pos = atom_pos_all[:, 1]

            # get inter atoms
            res_mask = torch.zeros(atom_type_all.shape).bool()
            res_mask[:, 1] = True
            if self.args.residue_only:
                atom_masks_all *= res_mask
                assert torch.all(atom_masks_all.sum(-1) == 1)
            res_mask = res_mask.reshape(-1)

            inter_mask = atom_masks_all.clone().reshape(-1)

            n_res_all = atom_pos_all.shape[0]
            atom_pos_all_origin = atom_pos_all_origin.reshape(-1, 3)[inter_mask]
            atom_pos_all = atom_pos_all.reshape(-1, 3)[inter_mask]
            atom_type_all = atom_type_all.reshape(-1)[inter_mask]
            residue_type_origin = residue_type_origin.reshape(-1)[inter_mask]
            atom_type_origin = atom_type_origin.reshape(-1)[inter_mask]
            residue_type_all = residue_type_all.reshape(-1)[inter_mask]
            atom_pos_pred_index = atom_pos_pred_index.reshape(-1)[inter_mask]
            res_mask = res_mask[inter_mask]
            residue_pos_all = residue_pos_all.reshape(-1)[inter_mask]
            if self.args.use_sas or self.args.sas_pred:
                sas_res = sas[:, 1]
                sas_atom = sas.reshape(-1)[inter_mask]
                if self.args.use_virtual_node:
                    sas_res = torch.cat([sas_res, torch.zeros(1)], dim=0)
                    sas_atom = torch.cat([sas_atom, torch.zeros(1)], dim=0)
                if self.args.num_virtual_point > 0 and self.args.use_virtual_node:
                    for i in range(self.args.num_virtual_point):
                        sas_atom = torch.cat([sas_atom, torch.zeros(1)], dim=0)
            n_atom_all = atom_pos_all.shape[0]
            
            ###################################################
            distance_mat = distance_matrix(atom_pos, atom_pos)
            # knn = min(self.knn, distance_mat.shape[0] - 1)
            knn_res = min(self.args.knn_res, distance_mat.shape[0] - 1)
            
            assert knn_res > 0, (idx, distance_mat.shape)

            D_adjust = torch.from_numpy(distance_mat)
            _, e = torch.topk(D_adjust, min(knn_res, D_adjust.shape[-1]), dim=-1, largest=False)
            
            cols = e[:, :knn_res].reshape(-1)
            rows = np.arange(distance_mat.shape[0]).reshape(distance_mat.shape[0], 1)
            rows = np.tile(rows, (1, knn_res)).reshape(-1)
                
            assert cols.shape == rows.shape, (cols.shape, rows.shape)
            assert cols.shape[0] > 1, idx
            num_real_edges = cols.shape[0]
            ###################################################
            if self.args.num_virtual_point > 0:
                def calculate_equal_partitions(number, n):
                    partition_size = number // n
                    partitions = [partition_size * i for i in range(1, n)]
                    return partitions
                points = calculate_equal_partitions(atom_pos.shape[0], self.args.num_virtual_point + 1)
                mean_pos = []
                start = 0
                for point in points:
                    mean_pos.append(atom_pos[start:point].mean(dim=0))
                    start = point

            ###################################################
            if not self.args.residue_only:
                distance_mat_aa = distance_matrix(atom_pos_all, atom_pos_all)
                knn_atom = min(self.args.knn_atom2atom, distance_mat_aa.shape[0] - 1)
                assert knn_atom > 2, idx
                sorted_indices = np.argsort(distance_mat_aa, axis=-1)
                cols_aa = sorted_indices[:, :knn_atom].reshape(-1)
                rows_aa = np.arange(distance_mat_aa.shape[0]).reshape(distance_mat_aa.shape[0], 1)
                rows_aa = np.tile(rows_aa, (1, knn_atom)).reshape(-1)
            ###################################################

            if self.args.use_virtual_node:
                num_real = atom_pos_all.shape[0]
                if not self.args.residue_only:
                    rows_aa = np.concatenate([rows_aa, np.ones(num_real) * num_real], axis=0).astype(np.int32)
                    cols_aa = np.concatenate([cols_aa, np.arange(num_real)], axis=0).astype(np.int32)
                    rows_aa = np.concatenate([rows_aa, np.arange(num_real)], axis=0).astype(np.int32)
                    cols_aa = np.concatenate([cols_aa, np.ones(num_real) * num_real], axis=0).astype(np.int32)
                # else:
                #     rows_aa = np.ones(num_real) * num_real
                #     cols_aa = np.arange(num_real)
                #     rows_aa = np.concatenate([rows_aa, np.arange(num_real)], axis=0).astype(np.int32)
                #     cols_aa = np.concatenate([cols_aa, np.ones(num_real) * num_real], axis=0).astype(np.int32)
                atom_pos_all = np.concatenate([atom_pos_all, np.zeros((1, 3))], axis=0)
                atom_pos_all = torch.from_numpy(atom_pos_all)
                residue_type_all = np.concatenate([residue_type_all, np.ones(1) * (23 + data_flag)], axis=0)
                atom_type_all = np.concatenate([atom_type_all, np.ones(1) * (38 + data_flag)], axis=0)
                atom_pos_pred_index = torch.from_numpy(np.concatenate([atom_pos_pred_index, np.zeros(1)], axis=0))
                atom_type_origin = torch.from_numpy(np.concatenate([atom_type_origin, np.ones(1) * (38 + data_flag)], axis=0))
                cols = torch.cat([cols, torch.ones(n_res_all) * n_res_all], dim=0)
                rows = torch.cat([torch.from_numpy(rows), torch.arange(n_res_all)], dim=0)
                rows = torch.cat([rows, torch.ones(n_res_all) * n_res_all], dim=0)
                cols = torch.cat([cols, torch.arange(n_res_all)], dim=0)
                atom_pos = torch.cat([atom_pos, torch.zeros(1, 3)], dim=0)
                residue_pos_all = torch.cat([residue_pos_all, torch.ones(1).long() * n_res_all])
                atom_pos_all_origin = torch.cat([atom_pos_all_origin, torch.zeros(1, 3)], dim=0)
                residue_type_origin = torch.cat([residue_type_origin, torch.ones(1) * (23 + data_flag)])
                res_mask = torch.cat([res_mask, torch.ones(1).bool()])

            if self.args.num_virtual_point > 0 and self.args.use_virtual_node:

                for i in range(self.args.num_virtual_point):
                    num_real_res_ = n_res_all + i + 1
                    # if not self.args.residue_only:
                    #     rows_aa = np.concatenate([rows_aa, np.ones(num_real) * num_real], axis=0).astype(np.int32)
                    #     cols_aa = np.concatenate([cols_aa, np.arange(num_real)], axis=0).astype(np.int32)
                    #     rows_aa = np.concatenate([rows_aa, np.arange(num_real)], axis=0).astype(np.int32)
                    #     cols_aa = np.concatenate([cols_aa, np.ones(num_real) * num_real], axis=0).astype(np.int32)
                    # else:
                    #     rows_aa = np.ones(num_real) * num_real
                    #     cols_aa = np.arange(num_real)
                    #     rows_aa = np.concatenate([rows_aa, np.arange(num_real)], axis=0).astype(np.int32)
                    #     cols_aa = np.concatenate([cols_aa, np.ones(num_real) * num_real], axis=0).astype(np.int32)
                    # print(atom_pos_all.shape, mean_pos[i].shape)
                    atom_pos_all = np.concatenate([atom_pos_all, mean_pos[i].unsqueeze(0).numpy()], axis=0)
                    atom_pos_all = torch.from_numpy(atom_pos_all)
                    residue_type_all = np.concatenate([residue_type_all, np.ones(1) * (24 + data_flag)], axis=0)
                    atom_type_all = np.concatenate([atom_type_all, np.ones(1) * (39 + data_flag)], axis=0)
                    atom_pos_pred_index = torch.from_numpy(np.concatenate([atom_pos_pred_index, np.zeros(1)], axis=0))
                    atom_type_origin = torch.from_numpy(np.concatenate([atom_type_origin, np.ones(1) * (39 + data_flag)], axis=0))
                    cols = torch.cat([cols, torch.ones(num_real_res_) * num_real_res_], dim=0)
                    rows = torch.cat([rows, torch.arange(num_real_res_)], dim=0)
                    rows = torch.cat([rows, torch.ones(num_real_res_) * num_real_res_], dim=0)
                    cols = torch.cat([cols, torch.arange(num_real_res_)], dim=0)
                    atom_pos = torch.cat([atom_pos, mean_pos[i].unsqueeze(0)], dim=0)
                    residue_pos_all = torch.cat([residue_pos_all, torch.ones(1).long() * num_real_res_])
                    atom_pos_all_origin = torch.cat([atom_pos_all_origin, mean_pos[i].unsqueeze(0)], dim=0)
                    residue_type_origin = torch.cat([residue_type_origin, torch.ones(1) * (24 + data_flag)])
                    res_mask = torch.cat([res_mask, torch.ones(1).bool()])
                
            try:
                rows = torch.from_numpy(rows)
            except:
                pass
            rows = rows.long()
            cols = cols.long()
            edge_vecs = []
            if self.args.use_absolute:
                edge_vecs.append(self.compute_sin_cos(atom_pos[rows] - atom_pos[cols]))
            if self.args.use_relative:
                edge_vecs.append(self.compute_sin_cos_res(knn_res, atom_pos[rows] , atom_pos[cols], atom_pos_all37[:, [0, 1, 2, 4]],True, num_real_edges))
            edge_vec = torch.cat(edge_vecs, dim=-1)
            cols = cols.unsqueeze(-1).long()
            rows = rows.unsqueeze(-1).long()
            edge_index = torch.cat([rows, cols], dim=-1)
            edge_vec = torch.cat(edge_vecs, dim=-1)

            if not self.args.residue_only:
                rows_aa = torch.from_numpy(rows_aa)
                cols_aa = torch.from_numpy(cols_aa)
                edge_aa_vecs = []

                if self.args.use_absolute:
                    edge_aa_vecs.append(self.compute_sin_cos(atom_pos_all[rows_aa] - atom_pos_all[cols_aa]))
                if self.args.use_relative:
                    edge_aa_vecs.append(self.compute_sin_cos_aa(knn_atom, atom_pos_all[rows_aa], rows_aa, atom_pos_all[cols_aa], cols_aa, residue_pos_all, atom_pos_all37[:, [0, 1, 2, 4]], True))
                edge_aa_vec = torch.cat(edge_aa_vecs, dim=-1)
                # edge_aa_vec = compute_sin_cos(atom_pos_all[rows_aa] - atom_pos_all[cols_aa])
                rows_aa = rows_aa.unsqueeze(-1)
                cols_aa = cols_aa.unsqueeze(-1)
                aa_edge_index = torch.cat([rows_aa, cols_aa], dim=-1)

                assert rows.shape[0] > 2, idx
                assert rows_aa.shape[0] > 2, idx
            else:
                edge_aa_vecs = None
                aa_edge_index = None
            assert torch.all(atom_type_origin != 37)
            assert torch.all(residue_type_origin != 22)

            assert atom_type_all.shape[0] == atom_pos_pred_index.shape[0] == atom_pos_all.shape[0] == residue_type_all.shape[0] == res_mask.shape[0]
            assert atom_pos_all.shape[0] > 2, idx
            assert edge_index.shape[0] > 2, idx
            assert res_mask.sum() > 0, idx
            is_train = torch.Tensor([self.is_train * 1])
            batch_index_res = torch.ones(res_mask.sum() - 1)
            batch_id = torch.ones(res_mask.sum())
            # print_all_shape(torsion, torsion_mask)
            if self.args.use_virtual_node and self.args.use_esm_feat:
                esm_feat = torch.cat([esm_feat, esm_cls_feat.unsqueeze(0)])
                for i in range(self.args.num_virtual_point):
                    esm_feat = torch.cat([esm_feat, esm_cls_feat.unsqueeze(0)])
                assert esm_feat.shape[0] == atom_pos.shape[0]

            return {
                "atom_pos": atom_pos_all,
                "residue_pos": (residue_pos.long(), atom_pos.shape[0]),
                "residue_pos_all": (residue_pos_all.long(), atom_pos.shape[0]),
                "atom_type": torch.from_numpy(atom_type_all.reshape(-1)).long(),
                "residue_type": torch.from_numpy(residue_type_all).long(),
                "edge_index": (edge_index, atom_pos.shape[0]),
                "aa_edge_index": (aa_edge_index.long(), atom_type_all.shape[0]) if not self.args.residue_only else None,
                "atom_pos_all_origin": atom_pos_all_origin,
                "residue_type_origin": residue_type_origin,
                "atom_type_origin": atom_type_origin,
                "atom_pos_pred_index": atom_pos_pred_index,
                "batch_index": torch.ones(atom_pos_all.shape[0]),
                "batch_index_res": batch_index_res,
                "batch_id": batch_id,
                "esm_feat": esm_feat,
                "edge_vec": edge_vec,
                "edge_aa_vec": edge_aa_vec if not self.args.residue_only else None,
                "torsion": torch.from_numpy(torsion).reshape(torsion.shape[0], -1), 
                "torsion_mask": torch.from_numpy(torsion_mask).unsqueeze(-1).expand(-1, -1, 2).reshape(torsion.shape[0], -1),
                "res_mask": res_mask.bool(),
                "idx": torch.Tensor([idx]),
                "is_train": is_train,
                "atom_sas": sas_atom if self.args.use_sas or self.args.sas_pred else None,
            }



class InferenceUniProtDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(self, dataset, args, split, crop_rational=1/4, is_train=False):
        super().__init__(dataset)
        self.dataset = dataset
        self.is_train = is_train
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.set_epoch(None)
        self.keep = args.keep
        self.crop = args.crop
        # rotate sample
        # current solution: randomly sample rotation
        self.n_sample_random = 4
        self.knn = args.knn
        self.num_rbf = 16
        self.split = split

        self.funcs = [torch.sin, torch.cos]
        self.xyz_N_freqs = 10
        self.xyz_freq_bands = 2**torch.linspace(0, self.xyz_N_freqs-1, self.xyz_N_freqs)
        self.theta_N_freqs = 4
        self.theta_freq_bands = 2**torch.linspace(0, self.theta_N_freqs-1, self.theta_N_freqs)


    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def GetRotationMatrix(self, theta, phi, gamma):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)], 
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)], 
                       [0, 1, 0],  
                       [-np.sin(phi), 0, np.cos(phi)]])        
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])       
        return np.dot(Rz, np.dot(Ry, Rx))

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    def sample_trans(self, before, rotate_op, shift_op, scale_op):
        ss = before * scale_op.reshape(1, 3) + shift_op.reshape(-1, 3)
        after = rotate_op.T @ ss.transpose(1, 0)
        return after

    def GetRotationMatrix(self, theta, phi, gamma):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)], 
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)], 
                       [0, 1, 0],  
                       [-np.sin(phi), 0, np.cos(phi)]])        
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])       
        return np.dot(Rz, np.dot(Ry, Rx))
    
    def get_nerf_feat(self, x, freq_bands):
        out = [x]
        for freq in freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.stack(out, -1)

    def get_random_fourier_feat(self, coords):
        assert self.rand_freqs is not None
        # k = coords . rand_freqs
        # expand rand_freqs with singleton dimension along the batch dimensions
        # e.g. dim (1, ..., 1, n_rand_feats, 3)
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2

        kxkykz = coords[..., None, 0:3] * freqs  # compute the x,y,z components of k
        k = kxkykz.sum(-1)  # compute k
        s = torch.sin(k)
        c = torch.cos(k)
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        print(x.shape, coords.shape)
        return x

    def compute_sin_cos(self, vectors):
        angle_xy = torch.atan2(torch.norm(vectors[:, 2:3], dim=1), torch.norm(vectors[:, :2], dim=1))
        sin_xy = torch.sin(angle_xy)
        cos_xy = torch.cos(angle_xy)
        
        angle_xz = torch.atan2(torch.norm(vectors[:, 1:2], dim=1), torch.norm(vectors[:, [0, 2]], dim=1))
        sin_xz = torch.sin(angle_xz)
        cos_xz = torch.cos(angle_xz)
        
        angle_yz = torch.atan2(torch.norm(vectors[:, :2], dim=1), torch.norm(vectors[:, [1, 2]], dim=1))
        sin_yz = torch.sin(angle_yz)
        cos_yz = torch.cos(angle_yz)
        
        if not self.args.use_nerf_encoding:
            result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)
        elif not self.args.random_fourier:
            result = []
            result.append(self.get_nerf_feat(torch.atan2(sin_xy, cos_xy), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_xz, cos_xz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_yz, cos_yz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(vectors, self.xyz_freq_bands).reshape(-1, 63))
            result = torch.cat(result, dim=-1)
        else:
            assert 0
            result = []
            result.append(self.get_random_fourier_feat(torch.atan2(sin_xy, cos_xy), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_xz, cos_xz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_yz, cos_yz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.norm(vectors, dim=-1, keepdim=True), self.xyz_freq_bands).reshape(-1, 63))
            result = torch.cat(result, dim=-1)
        return result

    def compute_sin_cos_res(self, knn,start,dest,X,use_local_coord):
        vectors = dest-start
        if use_local_coord:
            # use true atoms only
            num_real = X.shape[0]
            num = vectors.shape[0]
            real_vec = vectors[:-2 * num_real,:]
            assert real_vec.shape[0] == num-2*num_real

            X = X.unsqueeze(0)
            V = X.clone()
            X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
            dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
            U = _normalize(dX, dim=-1)
            u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
            n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
            b_1 = _normalize(u_0 - u_1, dim=-1)

            n_0 = n_0[:,::3,:]
            b_1 = b_1[:,::3,:]
            if b_1.shape[1] >= real_vec.shape[0]//knn:
                assert 0
                err = b_1.shape[1] - real_vec.shape[0]//knn+1
                n_0 = n_0[:,:-err,:]
                b_1 = b_1[:,:-err,:]
            X = X[:,::3,:] #[B,N,3]
            Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
            Q = Q.view(list(Q.shape[:2]) + [9]) # [B, N, 3,3]
            Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [B, N, 9]
            Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2) # [B, N, 1, 3, 3]

            real_vec = real_vec.reshape(-1,knn,3).unsqueeze(0).unsqueeze(3) # [B,N,30,1,3]
            dU = torch.matmul(Q[:,:,:,None,:,:], real_vec[...,None]).squeeze(-1) # [B, N, 30,1,3] 邻居的相对坐标
            B, N, K = dU.shape[:3]
            E_direct = _normalize(dU, dim=-1)
            E_direct = E_direct.reshape(N*K,-1)
            vectors = torch.cat((E_direct,vectors[num - 2 * num_real:,:]),dim=0)

        angle_xy = torch.atan2(torch.norm(vectors[:, 2:3], dim=1), torch.norm(vectors[:, :2], dim=1))
        sin_xy = torch.sin(angle_xy)
        cos_xy = torch.cos(angle_xy)

        angle_xz = torch.atan2(torch.norm(vectors[:, 1:2], dim=1), torch.norm(vectors[:, [0, 2]], dim=1))
        sin_xz = torch.sin(angle_xz)
        cos_xz = torch.cos(angle_xz)

        angle_yz = torch.atan2(torch.norm(vectors[:, :2], dim=1), torch.norm(vectors[:, [1, 2]], dim=1))
        sin_yz = torch.sin(angle_yz)
        cos_yz = torch.cos(angle_yz)

        if not self.args.use_nerf_encoding:
            result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)
        else:
            result = []
            result.append(self.get_nerf_feat(torch.atan2(sin_xy, cos_xy), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_xz, cos_xz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_yz, cos_yz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(vectors, self.xyz_freq_bands).reshape(-1, 63))
            result = torch.cat(result, dim=-1)
        return result


    def compute_sin_cos_aa(self, knn,start,rows_aa,dest,cols_aa,residue_pos_all,X,use_local_coord):
        vectors = dest-start
        if use_local_coord:
            num_real = X.shape[0]
            num = vectors.shape[0]
            step = num_real * 32
            # num_remove = num_real * 2
            if self.args.use_virtual_node:
                X = torch.cat([X, torch.zeros((1, 4, 3))], axis=0)
            # # use true atoms only
            # num_segments = vectors.shape[0] // step
            # segments = np.split(vectors, num_segments)
            # trimmed_segments = [segment[:-num_remove, :] for segment in segments]
            # real_vec = torch.tensor(np.concatenate(trimmed_segments, axis=0))

            X = X.unsqueeze(0)
            V = X.clone()
            X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
            dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
            U = _normalize(dX, dim=-1)
            u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
            n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
            b_1 = _normalize(u_0 - u_1, dim=-1)

            n_0 = n_0[:,::3,:]
            b_1 = b_1[:,::3,:]
            # if b_1.shape[1] >= real_vec.shape[0]//knn:
            #     err = b_1.shape[1] - real_vec.shape[0]//knn+1
            #     n_0 = n_0[:,:-err,:]
            #     b_1 = b_1[:,:-err,:]
            X = X[:,::3,:] #[B,N,3]
            Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
            Q = Q.view(list(Q.shape[:2]) + [9]) # [B, N, 3,3]
            Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [B, N, 9]
            Q = Q.view(list(Q.shape[:2]) + [3,3]) # [B, N, 3, 3]

            dU = torch.matmul(Q[:,residue_pos_all[rows_aa],:,:], vectors[...,None]).squeeze(-1) # [B,N,30,4,3] 邻居的相对坐标
            # B, N, K = dU.shape[:3]
            E_direct = _normalize(dU, dim=-1)
            vectors = E_direct.squeeze(0)
            # E_direct = E_direct.reshape(N*K*4,-1)
            # vir_segments = [segment[num_real * knn:, :] for segment in segments]
            # real_segments = np.split(E_direct, num_segments)
            # trimmed_segments = [torch.cat((a,b),dim=0) for a,b in zip(vir_segments,real_segments)]
            # vectors = torch.tensor(np.concatenate(trimmed_segments, axis=0))

        angle_xy = torch.atan2(torch.norm(vectors[:, 2:3], dim=1), torch.norm(vectors[:, :2], dim=1))
        sin_xy = torch.sin(angle_xy)
        cos_xy = torch.cos(angle_xy)

        angle_xz = torch.atan2(torch.norm(vectors[:, 1:2], dim=1), torch.norm(vectors[:, [0, 2]], dim=1))
        sin_xz = torch.sin(angle_xz)
        cos_xz = torch.cos(angle_xz)

        angle_yz = torch.atan2(torch.norm(vectors[:, :2], dim=1), torch.norm(vectors[:, [1, 2]], dim=1))
        sin_yz = torch.sin(angle_yz)
        cos_yz = torch.cos(angle_yz)

        result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)

        if not self.args.use_nerf_encoding:
            result = torch.stack([sin_xy, cos_xy, sin_xz, cos_xz, sin_yz, cos_yz], dim=1)
        else:
            result = []
            result.append(self.get_nerf_feat(torch.atan2(sin_xy, cos_xy), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_xz, cos_xz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_yz, cos_yz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(vectors, self.xyz_freq_bands).reshape(-1, 63))
            result = torch.cat(result, dim=-1)
        return result

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):
            data = self.dataset[idx]
            try:
                data = gzip.decompress(data)
            except:
                pass
            try:
                data = pickle.loads(data)
            except:
                pass
            '''
            # atom_pred_pos_index
            # atom_pos
            # atom_type
            # residue_type
            # edge_weight
            # edge_res_attr
            # edge_atom_attr
            # edge_index
            # edge_diff
            # atom_pos_origin
            # residue_type_origin
            # atom_type_origin
            '''
            is_nan = np.isnan(np.sum(data["atom_pos"], axis=(1, 2)))
            mask_nan = ~is_nan
            data["atom_pos"] = data["atom_pos"][mask_nan]
            data["atom_masks"] = data["atom_masks"][mask_nan]
            data["residue_type"] = data["residue_type"][mask_nan]
            eps = 1e-12
            rdm = np.random.uniform(0, 1, size=(1))
            mode_3d = True
            mode_esm = True
            if rdm <= 0.5:
                mode_3d = True
                mode_esm = True
            elif rdm < 0.75:
                mode_3d = False
            else:
                mode_esm = False

            try:
                data_flag = data["dataset"]
                data_flag = 0
                # data["atom_pos"] = data["atom_pos"] + np.random.randn(*data["atom_pos"].shape) * 0.1
            except:
                data_flag = 0

            torsion, torsion_mask = atom37_to_torsion_angles(data["residue_type"], data["atom_pos"], data["atom_masks"])

            num_res = data["residue_type"].shape[0]
            residue_type_all = torch.from_numpy(data["residue_type"]).unsqueeze(-1).expand(-1, 37).long()
            atom_masks_all = torch.from_numpy(data["atom_masks"]).bool()
            atom_pos_all = torch.from_numpy(data["atom_pos"])
            atom_type_all = torch.from_numpy(np.arange(37)).unsqueeze(0).expand(num_res, -1)
            assert torch.all(atom_masks_all[:, 1])
            esm_feat = data["esm_feat"].half()
            esm_cls_feat = data["esm_feat_cls"].half()
            pdb_id = data["id"]
            # mask if no alpha C
            # C_mask = torch.zeros(atom_masks_all.shape).bool()
            # C_mask[:, 1] = True
            # C_mask = atom_masks_all.long() & C_mask
            # atom_pos_all = atom_pos_all[C_mask[:, 1].bool(), :, :]
            # atom_type_all = atom_type_all[C_mask[:, 1].bool(), :]
            # residue_type_all = residue_type_all[C_mask[:, 1].bool(), :]
            # atom_masks_all = atom_masks_all[C_mask[:, 1].bool(), :]
            # esm_feat = esm_feat[C_mask[:, 1].bool(), :]
            # torsion, torsion_mask = torsion[C_mask[:, 1].bool(), :, :], torsion_mask[C_mask[:, 1].bool(), :]

            # crop            
            # length_threshold = self.args.cutoff
            # if atom_pos_all.shape[0] > length_threshold:
            #     residue_type_all = residue_type_all[:length_threshold]
            #     atom_type_all = atom_type_all[:length_threshold]
            #     atom_pos_all = atom_pos_all[:length_threshold, :]
            #     atom_masks_all = atom_masks_all[:length_threshold, :]
            #     esm_feat = esm_feat[:length_threshold, :]
            #     torsion, torsion_mask = torsion[:length_threshold, :], torsion_mask[:length_threshold, :]
            residue_pos = torch.from_numpy(np.arange(atom_pos_all.shape[0]))
            residue_pos_all = torch.from_numpy(np.arange(atom_pos_all.shape[0])).unsqueeze(1).expand(-1, 37)
            theta, phi, gamma = np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)

            rot = self.GetRotationMatrix(theta, phi, gamma)
            trans = np.random.uniform(-20, 20, size=3)
            atom_pos_all = torch.from_numpy(np.dot(atom_pos_all.reshape(-1, 3), rot) + trans).reshape(-1, 37, 3)

            torsion2, torsion_mask = atom37_to_torsion_angles(data["residue_type"], np.array(atom_pos_all), data["atom_masks"])

            assert np.sum(torsion - torsion2) < 1, (torsion, torsion2) 

            atom_pos_all = remove_center(atom_pos_all, atom_masks_all, eps)
            # save for label
            atom_pos_all_origin = atom_pos_all.clone().reshape(-1, 3)
            residue_type_origin = residue_type_all.clone().reshape(-1)

            # add noise
            n_atom = atom_pos_all.shape[0]
            atom_type_origin = atom_type_all.clone()
            
            # only save alpahc 
            if self.args.sas_pred:
                import freesasa
                radius_dic = {
                    'N': 1.6,
                    'C': 1.7,
                    'O': 1.4,
                    'S': 1.8,
                }
                atom_radius = torch.Tensor([radius_dic[a[0]] for a in atom_types])
                atoms_ = atom_pos_all.reshape(-1, 3)
                atoms_m_ = atom_masks_all.reshape(-1).bool()
                atoms_r_ = np.tile(atom_radius.reshape(1, 37), (atom_pos_all.shape[0], 1)).reshape(-1)
                sas = torch.zeros(atoms_.shape[0])
                atoms_id_ = torch.arange(atoms_.shape[0])
                atoms_ = atoms_[atoms_m_].reshape(-1)
                atoms_id_ = atoms_id_[atoms_m_]
                atoms_r_ = atoms_r_[atoms_m_]
                sas_atom = freesasa.calcCoord(atoms_, atoms_r_)
                for id_s in range(len(atoms_id_)):
                    sas[atoms_id_[id_s]] = sas_atom.atomArea(id_s)
                sas = sas.reshape(-1, 37)

            # res_type_mask_prob
            # res_noise_scale
            select_num = int(self.args.res_type_mask_prob * n_atom)
            index_select = np.random.choice(np.arange(n_atom), select_num, replace=False)
            # residue_type_all[index_select] = torch.ones(len(index_select), 37, dtype=torch.long) * 22
            # esm_feat[index_select] = torch.zeros(len(index_select), esm_feat.shape[1], dtype=torch.long).type(torch.half)
            # if self.args.mask_side_chain:
            #     side_chain_mask = torch.zeros(len(index_select), 37, dtype=torch.long)
            #     side_chain_mask[:, [0, 1, 2, 4]] = 1
            #     atom_masks_all[index_select] = (atom_masks_all[index_select] * side_chain_mask).bool()

            # select_num = int(self.args.res_pos_mask_prob * n_atom)
            # if not self.args.mask_same:
            #     index_select = np.random.choice(np.arange(n_atom), select_num, replace=False)
            
            # atom_pos_all[index_select] += torch.from_numpy(np.random.randn(len(index_select), 37, 3)).float() * self.args.res_noise_scale
            atom_pos_pred_index = np.zeros((n_atom, 37))
            atom_pos_pred_index[index_select] += 1
            atom_pos_pred_index = torch.from_numpy(atom_pos_pred_index).long()

            atom_pos_all37 = atom_pos_all.clone()
            atom_masks_all37 = atom_masks_all.clone()
            # only save alpahc 
            if self.args.use_sas and not self.args.sas_pred:
                import freesasa
                radius_dic = {
                    'N': 1.6,
                    'C': 1.7,
                    'O': 1.4,
                    'S': 1.8,
                }
                atom_radius = torch.Tensor([radius_dic[a[0]] for a in atom_types])
                atoms_ = atom_pos_all.reshape(-1, 3)
                atoms_m_ = atom_masks_all.reshape(-1).bool()
                atoms_r_ = np.tile(atom_radius.reshape(1, 37), (atom_pos_all.shape[0], 1)).reshape(-1)
                sas = torch.zeros(atoms_.shape[0])
                atoms_id_ = torch.arange(atoms_.shape[0])
                atoms_ = atoms_[atoms_m_].reshape(-1)
                atoms_id_ = atoms_id_[atoms_m_]
                atoms_r_ = atoms_r_[atoms_m_]
                sas_atom = freesasa.calcCoord(atoms_, atoms_r_)
                for id_s in range(len(atoms_id_)):
                    sas[atoms_id_[id_s]] = sas_atom.atomArea(id_s)
                sas = sas.reshape(-1, 37)
        
            atom_pos = atom_pos_all[:, 1]

            # get inter atoms
            res_mask = torch.zeros(atom_type_all.shape).bool()
            res_mask[:, 1] = True
            res_mask = res_mask.reshape(-1)

            inter_mask = atom_masks_all.clone().reshape(-1)

            atom_pos_all_origin = atom_pos_all_origin.reshape(-1, 3)[inter_mask]
            atom_pos_all = atom_pos_all.reshape(-1, 3)[inter_mask]
            atom_type_all = atom_type_all.reshape(-1)[inter_mask]
            residue_type_origin = residue_type_origin.reshape(-1)[inter_mask]
            atom_type_origin = atom_type_origin.reshape(-1)[inter_mask]
            residue_type_all = residue_type_all.reshape(-1)[inter_mask]
            atom_pos_pred_index = atom_pos_pred_index.reshape(-1)[inter_mask]
            res_mask = res_mask[inter_mask]
            residue_pos_all = residue_pos_all.reshape(-1)[inter_mask]
            if self.args.use_sas or self.args.sas_pred:
                sas_res = sas[:, 1]
                sas_atom = sas.reshape(-1)[inter_mask]
                if self.args.use_virtual_node:
                    sas_res = torch.cat([sas_res, torch.zeros(1)], dim=0)
                    sas_atom = torch.cat([sas_atom, torch.zeros(1)], dim=0)
            n_atom_all = atom_pos_all.shape[0]
            
            ###################################################
            distance_mat = distance_matrix(atom_pos, atom_pos)
            # knn = min(self.knn, distance_mat.shape[0] - 1)
            knn_res = min(self.args.knn_res, distance_mat.shape[0] - 1)
            
            assert knn_res > 0, (idx, distance_mat.shape)

            D_adjust = torch.from_numpy(distance_mat)
            _, e = torch.topk(D_adjust, min(knn_res, D_adjust.shape[-1]), dim=-1, largest=False)
            
            cols = e[:, :knn_res].reshape(-1)
            rows = np.arange(distance_mat.shape[0]).reshape(distance_mat.shape[0], 1)
            rows = np.tile(rows, (1, knn_res)).reshape(-1)
                
            assert cols.shape == rows.shape, (cols.shape, rows.shape)
            assert cols.shape[0] > 1, idx

            ###################################################
            distance_mat_aa = distance_matrix(atom_pos_all, atom_pos_all)
            knn_atom = min(self.args.knn_atom2atom, distance_mat_aa.shape[0] - 1)
            assert knn_atom > 2, idx
            sorted_indices = np.argsort(distance_mat_aa, axis=-1)
            cols_aa = sorted_indices[:, :knn_atom].reshape(-1)
            rows_aa = np.arange(distance_mat_aa.shape[0]).reshape(distance_mat_aa.shape[0], 1)
            rows_aa = np.tile(rows_aa, (1, knn_atom)).reshape(-1)
            ###################################################

            if self.args.use_virtual_node:
                num_real = atom_pos_all.shape[0]
                rows_aa = np.concatenate([rows_aa, np.ones(num_real) * num_real], axis=0).astype(np.int32)
                cols_aa = np.concatenate([cols_aa, np.arange(num_real)], axis=0).astype(np.int32)
                rows_aa = np.concatenate([rows_aa, np.arange(num_real)], axis=0).astype(np.int32)
                cols_aa = np.concatenate([cols_aa, np.ones(num_real) * num_real], axis=0).astype(np.int32)
                atom_pos_all = np.concatenate([atom_pos_all, np.zeros((1, 3))], axis=0)
                atom_pos_all = torch.from_numpy(atom_pos_all)
                residue_type_all = np.concatenate([residue_type_all, np.ones(1) * (23 + data_flag)], axis=0)
                atom_type_all = np.concatenate([atom_type_all, np.ones(1) * (38 + data_flag)], axis=0)
                atom_pos_pred_index = torch.from_numpy(np.concatenate([atom_pos_pred_index, np.zeros(1)], axis=0))
                atom_type_origin = torch.from_numpy(np.concatenate([atom_type_origin, np.ones(1) * (38 + data_flag)], axis=0))

            assert cols_aa.shape == rows_aa.shape, (cols_aa.shape, rows_aa.shape)

            assert cols_aa.shape[0] > 0, idx
            atom_type_all = np.array(atom_type_all).astype(np.int64)
            residue_type_all = np.array(residue_type_all)

            if self.args.use_virtual_node:
                num_real = atom_pos.shape[0]
                cols = torch.cat([cols, torch.ones(num_real) * num_real], dim=0)
                rows = torch.cat([torch.from_numpy(rows), torch.arange(num_real)], dim=0)
                rows = torch.cat([rows, torch.ones(num_real) * num_real], dim=0)
                cols = torch.cat([cols, torch.arange(num_real)], dim=0)
                atom_pos = torch.cat([atom_pos, torch.zeros(1, 3)], dim=0)
                residue_pos_all = torch.cat([residue_pos_all, torch.ones(1).long() * num_real])
                atom_pos_all_origin = torch.cat([atom_pos_all_origin, torch.zeros(1, 3)], dim=0)
                residue_type_origin = torch.cat([residue_type_origin, torch.ones(1) * (23 + data_flag)])
                res_mask = torch.cat([res_mask, torch.ones(1).bool()])
                assert not self.args.use_ipa
            try:
                rows = torch.from_numpy(rows)
            except:
                pass
            rows = rows.long()
            cols = cols.long()
            edge_vecs = []

            if self.args.use_absolute:
                edge_vecs.append(self.compute_sin_cos(atom_pos[rows] - atom_pos[cols]))
            if self.args.use_relative:
                edge_vecs.append(self.compute_sin_cos_res(knn_res, atom_pos[rows] , atom_pos[cols], atom_pos_all37[:, [0, 1, 2, 4]],True))
            edge_vec = torch.cat(edge_vecs, dim=-1)
            cols = cols.unsqueeze(-1).long()
            rows = rows.unsqueeze(-1).long()
            edge_index = torch.cat([rows, cols], dim=-1)


            edge_vec = torch.cat(edge_vecs, dim=-1)
            rows_aa = torch.from_numpy(rows_aa)
            cols_aa = torch.from_numpy(cols_aa)
            edge_aa_vecs = []

            if self.args.use_absolute:
                edge_aa_vecs.append(self.compute_sin_cos(atom_pos_all[rows_aa] - atom_pos_all[cols_aa]))
            if self.args.use_relative:
                edge_aa_vecs.append(self.compute_sin_cos_aa(knn_atom, atom_pos_all[rows_aa], rows_aa, atom_pos_all[cols_aa], cols_aa, residue_pos_all, atom_pos_all37[:, [0, 1, 2, 4]], True))
            edge_aa_vec = torch.cat(edge_aa_vecs, dim=-1)
            # edge_aa_vec = compute_sin_cos(atom_pos_all[rows_aa] - atom_pos_all[cols_aa])
            rows_aa = rows_aa.unsqueeze(-1)
            cols_aa = cols_aa.unsqueeze(-1)
            aa_edge_index = torch.cat([rows_aa, cols_aa], dim=-1)

            assert rows.shape[0] > 2, idx
            assert rows_aa.shape[0] > 2, idx
            assert torch.all(atom_type_origin != 37)
            assert torch.all(residue_type_origin != 22)

            assert atom_type_all.shape[0] == atom_pos_pred_index.shape[0] == atom_pos_all.shape[0] == residue_type_all.shape[0] == res_mask.shape[0]
            assert atom_pos_all.shape[0] > 5, idx
            assert edge_index.shape[0] > 2, idx
            assert aa_edge_index.shape[0] > 5, idx
            assert res_mask.sum() > 0, idx
            is_train = torch.Tensor([self.is_train * 1])
            batch_index_res = torch.ones(res_mask.sum() - 1)
            batch_id = torch.ones(res_mask.sum())
            # print_all_shape(torsion, torsion_mask)
            if self.args.use_virtual_node and self.args.use_esm_feat:
                esm_feat = torch.cat([esm_feat, esm_cls_feat.unsqueeze(0)])
                assert esm_feat.shape[0] == atom_pos.shape[0]

            return {
                "atom_pos": atom_pos_all,
                "residue_pos": (residue_pos.long(), atom_pos.shape[0]),
                "residue_pos_all": (residue_pos_all.long(), atom_pos.shape[0]),
                "atom_type": torch.from_numpy(atom_type_all.reshape(-1)).long(),
                "residue_type": torch.from_numpy(residue_type_all).long(),
                "edge_index": (edge_index, atom_pos.shape[0]),
                "aa_edge_index": (aa_edge_index.long(), atom_type_all.shape[0]),
                "atom_pos_all_origin": atom_pos_all_origin,
                "residue_type_origin": residue_type_origin,
                "atom_type_origin": atom_type_origin,
                "atom_pos_pred_index": atom_pos_pred_index,
                "batch_index": torch.ones(atom_pos_all.shape[0]),
                "batch_index_res": batch_index_res,
                "batch_id": batch_id,
                "esm_feat": esm_feat,
                "edge_vec": edge_vec,
                "edge_aa_vec": edge_aa_vec,
                "torsion": torch.from_numpy(torsion).reshape(torsion.shape[0], -1), 
                "torsion_mask": torch.from_numpy(torsion_mask).unsqueeze(-1).expand(-1, -1, 2).reshape(torsion.shape[0], -1),
                "res_mask": res_mask.bool(),
                "idx": torch.Tensor([idx]),
                "is_train": is_train,
                "atom_sas": sas_atom if self.args.use_sas or self.args.sas_pred else None,
                "pdb_id": pdb_id
            }

def convert_to_single_emb(x, sizes):
    # [128, 128] 
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i in range(len(sizes)):
        assert (x[..., i] < sizes[i]).all()
        x[..., i] = x[..., i] + offset
        offset += sizes[i]
    return x




class AtomEdgeIndexDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="atom_edge_index",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)
    def collater(self, samples):
        total_length_row = 0
        total_length_col = 0
        new_samples = []
        for sample in samples:
            edge_index, row_length, col_length = sample
            edge_index_new = edge_index.clone()
            edge_index_new[:,0] += total_length_row
            edge_index_new[:,1] += total_length_col
            new_samples.append(edge_index_new)
            total_length_row += row_length
            total_length_col += col_length
        return torch.cat(new_samples, dim=0)

class DataFlagDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="data_flag",
    ):
        self.dataset = dataset
        self.dim = 1
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)


class StoreIdDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="store_id",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)

class AtomPosDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="atom_pos",
    ):
        self.dataset = dataset
        self.dim = 3
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.cat(samples, dim=0)



class ResidueDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="residue_type",
        pad=21
    ):
        self.dataset = dataset
        self.dim = 3
        self.key = key
        self.set_epoch(None)
        self.pad = pad

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.cat(samples, dim=0)

class AtomTypeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="atom_type",
    ):
        self.dataset = dataset
        self.dim = 3
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.cat(samples, dim=0)


class EdgeIndexDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="edge_index",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)
    def collater(self, samples):
        total_length = 0
        new_samples = []
        for sample in samples:
            edge_index, length = sample
            new_samples.append(edge_index + total_length)
            total_length = total_length + length
        return torch.cat(new_samples, dim=0)


class TriEdgeIndexDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="edge_index",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)
    def collater(self, samples):
        total_length = 0
        new_samples = []
        for sample in samples:
            edge_index, length = sample
            new_samples.append(edge_index + total_length)
            total_length += length
        try:
            return torch.cat(new_samples, dim=0)
        except:
            length = min([sample.shape[-1] for sample in new_samples])
            return torch.cat([sample[..., :length, :] for sample in new_samples])

class AtomEdgeAttrDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="atom_edge_attr",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)
    def collater(self, samples):
        return torch.cat(samples, dim=0)

class ResEdgeAttrDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="res_edge_attr",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)
    def collater(self, samples):
        return torch.cat(samples, dim=0)


class EdgeWeightDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="edge_weight",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)
    def collater(self, samples):
        return torch.cat(samples, dim=0)

class EdgeDiffDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="edge_diff",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)
    def collater(self, samples):
        return torch.cat(samples, dim=0)

class StringDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="pdb_id",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)
    def collater(self, samples):
        return samples


class BatchIndexDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="batch_index",
    ):
        self.dataset = dataset
        self.key = key
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        return self.dataset[index][self.key]

    def __len__(self):
        return len(self.dataset)
    def collater(self, samples):
        idx = 0
        for i in range(len(samples)):
            if (samples[i]==1).all():
                samples[i] = samples[i] * idx
            idx += 1
        return torch.cat(samples, dim=0)