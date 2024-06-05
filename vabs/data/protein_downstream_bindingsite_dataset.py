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
from ..models.frame import Frame, Rotation
try:
    import torchdrug
    from torchdrug.transforms import ProteinView
    from torchdrug.layers import functional
except:
    pass

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

element_position_map = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
}


res_types_list = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
]

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}

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

def compute_sin_cos_aa(knn,start,dest,residue_pos_all,X,use_local_coord):
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
        dU = torch.matmul(Q[:,residue_pos_all,:,:,:], real_vec[...,None]).squeeze(-1) # [B,N,30,4,3] 邻居的相对坐标
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



def print_all_shape(*tensors):
    print("shape>>", [_.shape for _ in tensors])

class siamdiffPocketDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(self, dataset, args, crop_rational=1/4, is_train=False):
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

    def rbf_vec(self, D, num_rbf=64):
        return D
        D_min, D_max, D_count = -20., 20., num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        rbf = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return rbf.reshape(D.shape[0], -1)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):
            data = self.dataset[idx]

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
            eps = 1e-12
            type = torch.float32

            num_res = data["residue_type"].shape[0]
            residue_type_all = torch.from_numpy(data["residue_type"]).unsqueeze(-1).expand(-1, 37).long()
            atom_masks_all = torch.from_numpy(data["atom_masks"]).bool()
            atom_pos_all = torch.from_numpy(data["atom_pos"]).type(type)
            atom_type_all = torch.from_numpy(np.arange(37)).unsqueeze(0).expand(num_res, -1)
            assert atom_pos_all.shape[0] > 0
            shape_mask = torch.ones(atom_masks_all.shape).long()
            try:
                pocket_label = torch.from_numpy(data["pocket_label"]) * shape_mask
            except:
                pocket_label = torch.from_numpy(data["pocket_label"]).unsqueeze(-1) * shape_mask
            pocket_label *= atom_masks_all
            pocket_label = pocket_label > 0
            # pocket_label = torch.any(pocket_label, dim=1)

            # mask if no alpha C
            C_mask = torch.zeros(atom_masks_all.shape).bool()
            C_mask[:, 1] = True
            C_mask = atom_masks_all.long() & C_mask
            atom_pos_all = atom_pos_all[C_mask[:, 1].bool(), :, :]
            atom_type_all = atom_type_all[C_mask[:, 1].bool(), :]
            residue_type_all = residue_type_all[C_mask[:, 1].bool(), :]
            atom_masks_all = atom_masks_all[C_mask[:, 1].bool(), :]
            pocket_label = pocket_label[C_mask[:, 1].bool()]

            assert atom_pos_all.shape[0] > 0

            length_threshold = self.args.cutoff
            if atom_pos_all.shape[0] > length_threshold and self.is_train:
                residue_type_all = residue_type_all[:length_threshold]
                atom_type_all = atom_type_all[:length_threshold]
                atom_pos_all = atom_pos_all[:length_threshold, :]
                atom_masks_all = atom_masks_all[:length_threshold, :]
                pocket_label = pocket_label[:length_threshold]
            n_res = atom_pos_all.shape[0]
            residue_pos = torch.from_numpy(np.arange(atom_pos_all.shape[0]))
            residue_pos_all = torch.from_numpy(np.arange(atom_pos_all.shape[0])).unsqueeze(1).expand(-1, 37)

            inter_mask = atom_masks_all.clone().reshape(-1).bool()
            res_mask = torch.zeros(atom_type_all.shape).bool()
            res_mask[:, 1] = True
            res_mask = atom_masks_all.reshape(-1)

            pocket_label = pocket_label.reshape(-1)[inter_mask]
            atom_pos_all = atom_pos_all.reshape(-1, 3)[inter_mask]
            atom_type_all = atom_type_all.reshape(-1)[inter_mask]
            residue_pos_all = residue_pos_all.reshape(-1)[inter_mask]
            residue_type_all = residue_type_all.reshape(-1)[inter_mask]
            res_mask = res_mask[inter_mask]


            node_position = atom_pos_all
            num_atom = node_position.shape[0]
            atom_type = torch.Tensor([element_position_map[atom_types[_][0]] for _ in atom_type_all])
            atom_name = [atom_types[_][0] for _ in atom_type_all]
            # atom_name = [torchdrug.data.Protein.atom_name2id.get(name, -1) for name in atom_name]
            atom2residue = residue_pos_all
            residue_type_name = [restype_1to3[res_types_list[_]] for _ in residue_type_all]
            residue_type = []
            residue_feature = []
            atom_feat = []
            lst_residue = -1
            for i in range(num_atom):
                if True:
                # if atom2residue[i] != lst_residue:
                    residue_type.append(torchdrug.data.Protein.residue2id.get(residue_type_name[i], 0))
                    residue_feature.append(torchdrug.data.feature.onehot(residue_type_name[i], torchdrug.data.feature.residue_vocab, allow_unknown=True))
                    atom_feat.append(torchdrug.data.feature.onehot(atom_name[i], torchdrug.data.feature.atom_vocab, allow_unknown=True))
                    lst_residue = atom2residue[i]
            residue_type = torch.as_tensor(residue_type)
            residue_feature = torch.as_tensor(residue_feature)
            atom_feat = torch.as_tensor(atom_feat)
            num_residue = residue_type.shape[0]

            edge_list = torch.as_tensor([[0, 0, 0]])
            bond_type = torch.as_tensor([0])
            atom_name = torch.as_tensor([torchdrug.data.Protein.atom_name2id.get(name, -1) for name in atom_name])

            protein = torchdrug.data.Protein(edge_list, atom_type, bond_type, num_node=num_atom, num_residue=num_residue,
                                   node_position=node_position, atom_name=atom_name,
                                    atom2residue=atom2residue, residue_feature=residue_feature, 
                                    residue_type=residue_type)
            # protein.view = 'residue'
            is_train = torch.Tensor([self.is_train * 1])

            atom_feature = torch.cat([
                residue_feature,
                atom_feat
            ], dim=-1)
            # protein.atom_feature = atom_feature
            # print(atom_feat.shape, pocket_label.shape)
            assert atom_feature.shape[0] == pocket_label.shape[0]
            return {
                "protein": protein,
                "pocket_label": pocket_label,
                "batch_index": torch.ones(atom_feature.shape[0]),
                "input": atom_feature.float(),
                "res_mask": res_mask,
            }

class GearInferenceDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(self, dataset, args, crop_rational=1/4, is_train=False):
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

    def rbf_vec(self, D, num_rbf=64):
        return D
        D_min, D_max, D_count = -20., 20., num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        rbf = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return rbf.reshape(D.shape[0], -1)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):
            data = self.dataset[idx]

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
            eps = 1e-12
            type = torch.float32

            num_res = data["residue_type"].shape[0]
            residue_type_all = torch.from_numpy(data["residue_type"]).unsqueeze(-1).expand(-1, 37).long()
            atom_masks_all = torch.from_numpy(data["atom_masks"]).bool()
            atom_pos_all = torch.from_numpy(data["atom_pos"]).type(type)
            pdb_id = data["id"]
            atom_type_all = torch.from_numpy(np.arange(37)).unsqueeze(0).expand(num_res, -1)
            assert atom_pos_all.shape[0] > 0
            shape_mask = torch.ones(atom_masks_all.shape).long()
            # try:
            #     pocket_label = torch.from_numpy(data["pocket_label"]) * shape_mask
            # except:
            #     pocket_label = torch.from_numpy(data["pocket_label"]).unsqueeze(-1) * shape_mask
            # pocket_label *= atom_masks_all
            # pocket_label = pocket_label > 0
            # pocket_label = torch.any(pocket_label, dim=1)

            # mask if no alpha C
            # C_mask = torch.zeros(atom_masks_all.shape).bool()
            # C_mask[:, 1] = True
            # C_mask = atom_masks_all.long() & C_mask
            # atom_pos_all = atom_pos_all[C_mask[:, 1].bool(), :, :]
            # atom_type_all = atom_type_all[C_mask[:, 1].bool(), :]
            # residue_type_all = residue_type_all[C_mask[:, 1].bool(), :]
            # atom_masks_all = atom_masks_all[C_mask[:, 1].bool(), :]
            # pocket_label = pocket_label[C_mask[:, 1].bool()]

            assert atom_pos_all.shape[0] > 0

            # length_threshold = self.args.cutoff
            # if atom_pos_all.shape[0] > length_threshold and self.is_train:
            #     residue_type_all = residue_type_all[:length_threshold]
            #     atom_type_all = atom_type_all[:length_threshold]
            #     atom_pos_all = atom_pos_all[:length_threshold, :]
            #     atom_masks_all = atom_masks_all[:length_threshold, :]
            #     pocket_label = pocket_label[:length_threshold]
            n_res = atom_pos_all.shape[0]
            residue_pos = torch.from_numpy(np.arange(atom_pos_all.shape[0]))
            residue_pos_all = torch.from_numpy(np.arange(atom_pos_all.shape[0])).unsqueeze(1).expand(-1, 37)

            inter_mask = atom_masks_all.clone().reshape(-1)

            res_mask = torch.zeros(atom_type_all.shape).bool()
            res_mask[:, 1] = True
            res_mask = res_mask.reshape(-1)

            atom_pos_all = atom_pos_all.reshape(-1, 3)[inter_mask]
            atom_type_all = atom_type_all.reshape(-1)[inter_mask]
            residue_pos_all = residue_pos_all.reshape(-1)[inter_mask]
            residue_type_all = residue_type_all.reshape(-1)[inter_mask]
            res_mask = res_mask[inter_mask]

            if not self.args.siam_type:
                node_position = atom_pos_all
                num_atom = node_position.shape[0]
                atom_type = torch.Tensor([element_position_map[atom_types[_][0]] for _ in atom_type_all])
                atom_name = [atom_types[_] for _ in atom_type_all]
                atom_name = torch.as_tensor([torchdrug.data.Protein.atom_name2id.get(name, -1) for name in atom_name])
                atom2residue = residue_pos_all
                residue_type_name = [restype_1to3[res_types_list[_]] for _ in residue_type_all]
                residue_type = []
                residue_feature = []
                lst_residue = -1
                for i in range(num_atom):
                    if atom2residue[i] != lst_residue:
                        residue_type.append(torchdrug.data.Protein.residue2id.get(residue_type_name[i], 0))
                        residue_feature.append(torchdrug.data.feature.onehot(residue_type_name[i], torchdrug.data.feature.residue_vocab, allow_unknown=True))
                        lst_residue = atom2residue[i]
                residue_type = torch.as_tensor(residue_type)
                residue_feature = torch.as_tensor(residue_feature)
                num_residue = residue_type.shape[0]
    
                edge_list = torch.as_tensor([[0, 0, 0]])
                bond_type = torch.as_tensor([0])
    
                protein = torchdrug.data.Protein(edge_list, atom_type, bond_type, num_node=num_atom, num_residue=num_residue,
                                       node_position=node_position, atom_name=atom_name,
                                        atom2residue=atom2residue, residue_feature=residue_feature, 
                                        residue_type=residue_type)
                protein.view = 'residue'
            else:    
                node_position = atom_pos_all
                num_atom = node_position.shape[0]
                atom_type = torch.Tensor([element_position_map[atom_types[_][0]] for _ in atom_type_all])
                atom_name = [atom_types[_][0] for _ in atom_type_all]
                atom2residue = residue_pos_all
                try:
                    residue_type_name = [restype_1to3[res_types_list[_]] for _ in residue_type_all]
                except:
                    print(torch.max(residue_type_all), len(res_types_list))
                    assert 0

                residue_type = []
                residue_feature = []
                atom_feat = []
                lst_residue = -1
                for i in range(num_atom):
                    if atom2residue[i] != lst_residue or self.args.siam_type:
                        residue_type.append(torchdrug.data.Protein.residue2id.get(residue_type_name[i], 0))
                        residue_feature.append(torchdrug.data.feature.onehot(residue_type_name[i], torchdrug.data.feature.residue_vocab, allow_unknown=True))
                        atom_feat.append(torchdrug.data.feature.onehot(atom_name[i], torchdrug.data.feature.atom_vocab, allow_unknown=True))
                        lst_residue = atom2residue[i]
                residue_type = torch.as_tensor(residue_type)
                residue_feature = torch.as_tensor(residue_feature)
                atom_feat = torch.as_tensor(atom_feat)
                num_residue = residue_type.shape[0]

                edge_list = torch.as_tensor([[0, 0, 0]])
                bond_type = torch.as_tensor([0])
                atom_name = torch.as_tensor([torchdrug.data.Protein.atom_name2id.get(name, -1) for name in atom_name])

                protein = torchdrug.data.Protein(edge_list, atom_type, bond_type, num_node=num_atom, num_residue=num_residue,
                                       node_position=node_position, atom_name=atom_name,
                                        atom2residue=atom2residue, residue_feature=residue_feature, 
                                        residue_type=residue_type)
                if not self.args.siam_type:
                    protein.view = 'residue'
                else:
                    residue_feature = torch.cat([
                        residue_feature,
                        atom_feat
                    ], dim=-1)
            is_train = torch.Tensor([self.is_train * 1])
            return {
                "protein": protein,
                # "pocket_label": pocket_label,
                "batch_index": torch.ones(n_res),
                "input": residue_feature.float(),
                "residue_pos_all": (residue_pos_all.long(), n_res),
                "residue_type": torch.from_numpy(data["residue_type"]).long(),
                "res_mask": res_mask.bool(),
                "atom_type": atom_type_all,
                "pdb_id": pdb_id
            }


class GearPocketDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(self, dataset, args, crop_rational=1/4, is_train=False):
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

    def rbf_vec(self, D, num_rbf=64):
        return D
        D_min, D_max, D_count = -20., 20., num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        rbf = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return rbf.reshape(D.shape[0], -1)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):
            data = self.dataset[idx]

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
            eps = 1e-12
            type = torch.float32

            num_res = data["residue_type"].shape[0]
            residue_type_all = torch.from_numpy(data["residue_type"]).unsqueeze(-1).expand(-1, 37).long()
            atom_masks_all = torch.from_numpy(data["atom_masks"]).bool()
            atom_pos_all = torch.from_numpy(data["atom_pos"]).type(type)
            atom_type_all = torch.from_numpy(np.arange(37)).unsqueeze(0).expand(num_res, -1)
            assert atom_pos_all.shape[0] > 0
            shape_mask = torch.ones(atom_masks_all.shape).long()
            try:
                pocket_label = torch.from_numpy(data["pocket_label"]) * shape_mask
            except:
                pocket_label = torch.from_numpy(data["pocket_label"]).unsqueeze(-1) * shape_mask
            pocket_label *= atom_masks_all
            pocket_label = pocket_label > 0
            pocket_label = torch.any(pocket_label, dim=1)

            # mask if no alpha C
            C_mask = torch.zeros(atom_masks_all.shape).bool()
            C_mask[:, 1] = True
            C_mask = atom_masks_all.long() & C_mask
            atom_pos_all = atom_pos_all[C_mask[:, 1].bool(), :, :]
            atom_type_all = atom_type_all[C_mask[:, 1].bool(), :]
            residue_type_all = residue_type_all[C_mask[:, 1].bool(), :]
            atom_masks_all = atom_masks_all[C_mask[:, 1].bool(), :]
            pocket_label = pocket_label[C_mask[:, 1].bool()]

            assert atom_pos_all.shape[0] > 0

            length_threshold = self.args.cutoff
            if atom_pos_all.shape[0] > length_threshold and self.is_train:
                residue_type_all = residue_type_all[:length_threshold]
                atom_type_all = atom_type_all[:length_threshold]
                atom_pos_all = atom_pos_all[:length_threshold, :]
                atom_masks_all = atom_masks_all[:length_threshold, :]
                pocket_label = pocket_label[:length_threshold]
            n_res = atom_pos_all.shape[0]
            residue_pos = torch.from_numpy(np.arange(atom_pos_all.shape[0]))
            residue_pos_all = torch.from_numpy(np.arange(atom_pos_all.shape[0])).unsqueeze(1).expand(-1, 37)

            inter_mask = atom_masks_all.clone().reshape(-1)

            res_mask = torch.zeros(atom_type_all.shape).bool()
            res_mask[:, 1] = True
            res_mask = res_mask.reshape(-1)

            atom_pos_all = atom_pos_all.reshape(-1, 3)[inter_mask]
            atom_type_all = atom_type_all.reshape(-1)[inter_mask]
            residue_pos_all = residue_pos_all.reshape(-1)[inter_mask]
            residue_type_all = residue_type_all.reshape(-1)[inter_mask]
            res_mask = res_mask[inter_mask]


            node_position = atom_pos_all
            num_atom = node_position.shape[0]
            atom_type = torch.Tensor([element_position_map[atom_types[_][0]] for _ in atom_type_all])
            atom_name = [atom_types[_] for _ in atom_type_all]
            atom_name = torch.as_tensor([torchdrug.data.Protein.atom_name2id.get(name, -1) for name in atom_name])
            atom2residue = residue_pos_all
            residue_type_name = [restype_1to3[res_types_list[_]] for _ in residue_type_all]
            residue_type = []
            residue_feature = []
            lst_residue = -1
            for i in range(num_atom):
                if atom2residue[i] != lst_residue:
                    residue_type.append(torchdrug.data.Protein.residue2id.get(residue_type_name[i], 0))
                    residue_feature.append(torchdrug.data.feature.onehot(residue_type_name[i], torchdrug.data.feature.residue_vocab, allow_unknown=True))
                    lst_residue = atom2residue[i]
            residue_type = torch.as_tensor(residue_type)
            residue_feature = torch.as_tensor(residue_feature)
            num_residue = residue_type.shape[0]

            edge_list = torch.as_tensor([[0, 0, 0]])
            bond_type = torch.as_tensor([0])

            protein = torchdrug.data.Protein(edge_list, atom_type, bond_type, num_node=num_atom, num_residue=num_residue,
                                   node_position=node_position, atom_name=atom_name,
                                    atom2residue=atom2residue, residue_feature=residue_feature, 
                                    residue_type=residue_type)
            protein.view = 'residue'
            is_train = torch.Tensor([self.is_train * 1])
            return {
                "protein": protein,
                "pocket_label": pocket_label,
                "batch_index": torch.ones(n_res),
                "input": residue_feature.float(),
                "res_mask": res_mask.bool(),
            }

restype_order_with_x = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20}
res_num2str = {v: k for k, v in restype_order_with_x.items()}

class ESMPocketDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(self, dataset, args, crop_rational=1/4, is_train=False):
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

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, idx):
        with data_utils.numpy_seed(self.args.seed, epoch, idx):
            data = self.dataset[idx]

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
            eps = 1e-12

            num_res = data["residue_type"].shape[0]
            residue_type_all = torch.from_numpy(data["residue_type"]).long()
            atom_masks_all = torch.from_numpy(data["atom_masks"]).bool()
            atom_pos_all = torch.from_numpy(data["atom_pos"])
            shape_mask = torch.ones(atom_masks_all.shape).long()
            try:
                pocket_label = torch.from_numpy(data["pocket_label"]) * shape_mask
            except:
                pocket_label = torch.from_numpy(data["pocket_label"]).unsqueeze(-1) * shape_mask
            pocket_label *= atom_masks_all
            pocket_label = pocket_label > 0
            pocket_label = torch.any(pocket_label, dim=1).unsqueeze(-1)

            # mask if no alpha C
            C_mask = torch.zeros(atom_masks_all.shape).bool()
            C_mask[:, 1] = True
            C_mask = atom_masks_all.long() & C_mask
            residue_type_all = residue_type_all[C_mask[:, 1].bool()]
            pocket_label = pocket_label[C_mask[:, 1].bool()]

            assert atom_pos_all.shape[0] > 0

            length_threshold = self.args.cutoff
            if atom_pos_all.shape[0] > length_threshold and self.is_train:
                residue_type_all = residue_type_all[:length_threshold]
                pocket_label = pocket_label[:length_threshold]

            residue_type_str = ""
            for idx in range(residue_type_all.shape[0]):
                num = residue_type_all[idx]
                residue_type_str += res_num2str[num.item()]

            return {
                "res_str": residue_type_str,
                "pocket_label_all": pocket_label,
                "batch_index": torch.ones(residue_type_all.shape[0]),
            }


class PocketDataset(BaseWrapperDataset):
    """A wrapper around a LMDB database that reads and returns items from it
    lazily."""

    def __init__(self, dataset, args, crop_rational=1/4, is_train=False):
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

    def get_nerf_feat(self, x, freq_bands):
        out = [x]
        for freq in freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.stack(out, -1)

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
        else:
            result = []
            result.append(self.get_nerf_feat(torch.atan2(sin_xy, cos_xy), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_xz, cos_xz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(torch.atan2(sin_yz, cos_yz), self.theta_freq_bands))
            result.append(self.get_nerf_feat(vectors, self.xyz_freq_bands).reshape(-1, 63))
            result = torch.cat(result, dim=-1)
        return result

    def compute_sin_cos_res(self, knn,start,dest,X,use_local_coord):
        vectors = dest-start
        if use_local_coord:
            # use true atoms only
            if self.args.use_virtual_node:
                num_real = X.shape[0]
                num = vectors.shape[0]
                real_vec = vectors[:-2 * num_real,:]
                assert real_vec.shape[0] == num-2*num_real
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
                vectors = torch.cat((E_direct,vectors[num - 2 * num_real:,:]),dim=0)
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
            eps = 1e-12

            num_res = data["residue_type"].shape[0]
            residue_type_all = torch.from_numpy(data["residue_type"]).unsqueeze(-1).expand(-1, 37).long()
            atom_masks_all = torch.from_numpy(data["atom_masks"]).bool()
            atom_pos_all = torch.from_numpy(data["atom_pos"])

            if self.args.atom_feat:
                atom_feat = torch.from_numpy(data["atom_feat"])
            else:
                atom_feat = torch.zeros(atom_pos_all.shape[0], 37, 5)

            if self.args.gb_feat:
                gb_feat_list = []
                if self.args.use_HMM:
                    gb_feat_list.append(data["HMM_feat"])
                if self.args.use_PSSM:
                    gb_feat_list.append(data["PSSM_feat"])
                if self.args.use_DSSP:
                    gb_feat_list.append(data["DSSP_feat"])
                gb_feat = torch.from_numpy(np.concatenate(gb_feat_list, axis=-1))
                # gb_feat = torch.from_numpy(np.concatenate([data["HMM_feat"], data["PSSM_feat"], data["DSSP_feat"]], axis=-1))
            else:
                gb_feat = None
            
            if self.args.use_ss:
                ss = torch.from_numpy(np.argmax(data["DSSP_feat"][:, :8], axis=-1)).long()
            else:
                ss = None
            assert atom_pos_all.shape[0] > 0
            # atom_pos = atom_pos_all.clone().reshape(-1, 3)
            shape_mask = torch.ones(atom_masks_all.shape).long()
            try:
                pocket_label = torch.from_numpy(data["pocket_label"]) * shape_mask
            except:
                pocket_label = torch.from_numpy(data["pocket_label"]).unsqueeze(-1) * shape_mask
            pocket_label *= atom_masks_all
            pocket_label = pocket_label > 0
            pocket_label = torch.any(pocket_label, dim=1).unsqueeze(-1).expand(-1, 37)
            pocket_label = pocket_label * atom_masks_all
            pocket_label = pocket_label > 0
            pocket_label_all = pocket_label.clone()
            atom_type_all = torch.from_numpy(np.arange(37)).unsqueeze(0).expand(num_res, -1)


            if self.args.use_esm_feat:
                esm_feat = data["esm_feat"].half()
            else:
                esm_feat = torch.zeros(atom_pos_all.shape[0], self.args.esm_dim).half()
                esm_cls_feat = torch.zeros(self.args.esm_dim).half()
            if self.args.use_esm_feat:
                try:
                    esm_cls_feat = data["esm_cls_feat"].half()
                except:
                    esm_cls_feat = data["esm_feat_cls"].half()
            else:
                esm_cls_feat = torch.zeros(self.args.esm_dim).half()

            # mask if no alpha C
            C_mask = torch.zeros(atom_masks_all.shape).bool()
            C_mask[:, 1] = True
            C_mask = atom_masks_all.long() & C_mask
            atom_feat = atom_feat[C_mask[:, 1].bool(), :, :]
            atom_pos_all = atom_pos_all[C_mask[:, 1].bool(), :, :]
            atom_type_all = atom_type_all[C_mask[:, 1].bool(), :]
            residue_type_all = residue_type_all[C_mask[:, 1].bool(), :]
            atom_masks_all = atom_masks_all[C_mask[:, 1].bool(), :]
            esm_feat = esm_feat[C_mask[:, 1].bool(), :]
            pocket_label = pocket_label[C_mask[:, 1].bool(), :]
            pocket_label_all = pocket_label_all[C_mask[:, 1].bool(), :]
            if gb_feat is not None:
                gb_feat = gb_feat[C_mask[:, 1].bool()]

            assert atom_pos_all.shape[0] > 0
            # if self.args.check_data:
            #     check_data_folder = r"/mnt/vepfs/users/zhaojiale/projs/check_data"
            #     def write_to_txt(filename, atom_pos1, atom_pos2, pocket_label1, pocket_label2):
            #         with open(filename, 'w') as file:
            #             for pos1, label1 in zip(atom_pos1, pocket_label1):
            #                 if label1 > 0:
            #                     file.write("N\t{:.6f}\t{:.6f}\t{:.6f}\n".format(pos1[0], pos1[1], pos1[2]))
            #                 else:
            #                     file.write("H\t{:.6f}\t{:.6f}\t{:.6f}\n".format(pos1[0], pos1[1], pos1[2]))
                        
            #             for pos2, label2 in zip(atom_pos2, pocket_label2):
            #                 if label2 > 0:
            #                     file.write("N\t{:.6f}\t{:.6f}\t{:.6f}\n".format(pos2[0], pos2[1], pos2[2]))
            #                 else:
            #                     file.write("H\t{:.6f}\t{:.6f}\t{:.6f}\n".format(pos2[0], pos2[1], pos2[2]))
            #     write_to_txt(os.path.join(check_data_folder, str(idx)+'_origin.xyz'), 
            #                  atom_pos,
            #                  atom_pos_all,
            #                  label,
            #                  pocket_label_all,
            #                  )

            # crop            
            length_threshold = self.args.cutoff
            if atom_pos_all.shape[0] > length_threshold and self.is_train:
                atom_feat = atom_feat[:length_threshold]
                residue_type_all = residue_type_all[:length_threshold]
                atom_type_all = atom_type_all[:length_threshold]
                atom_pos_all = atom_pos_all[:length_threshold, :]
                atom_masks_all = atom_masks_all[:length_threshold, :]
                esm_feat = esm_feat[:length_threshold, :]
                pocket_label = pocket_label[:length_threshold, :]
                pocket_label_all = pocket_label_all[:length_threshold, :]
                if gb_feat is not None:
                    gb_feat = gb_feat[:length_threshold]
            residue_pos = torch.from_numpy(np.arange(atom_pos_all.shape[0]))
            residue_pos_all = torch.from_numpy(np.arange(atom_pos_all.shape[0])).unsqueeze(1).expand(-1, 37)


            theta, phi, gamma = np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)

            rot = self.GetRotationMatrix(theta, phi, gamma)
            trans = np.random.uniform(-20, 20, size=3)
            atom_pos_all = torch.from_numpy(np.dot(atom_pos_all.reshape(-1, 3), rot) + trans).reshape(-1, 37, 3)
            atom_pos_all = remove_center(atom_pos_all, atom_masks_all, eps)
            # save for label
            atom_pos_origin = atom_pos_all[:, 1].clone()
            atom_pos_all_origin = atom_pos_all.clone()
            residue_type_origin = residue_type_all.clone()[:, 1]

            # add noise
            n_atom = atom_pos_all.shape[0]
            atom_type_origin = atom_type_all.clone()
            if self.is_train:
                select_num = int(self.args.res_type_mask_prob * n_atom)
                index_select = np.random.choice(np.arange(n_atom), select_num, replace=False)
                residue_type_all[index_select] = torch.ones(len(index_select), 37, dtype=torch.long) * 22
                atom_type_all[index_select] = torch.ones(len(index_select), 37, dtype=torch.long) * 37
                atom_pos_all[index_select] += torch.from_numpy(np.random.randn(len(index_select), 37, 3)).float() * self.args.res_noise_scale
                esm_feat[index_select] = torch.zeros(len(index_select), esm_feat.shape[1], dtype=torch.long).type(torch.half)
            if self.args.use_sas:
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
                assert torch.all(torch.from_numpy(atoms_r_) > 0)
                sas_atom = freesasa.calcCoord(atoms_, atoms_r_)
                for id_s in range(len(atoms_id_)):
                    sas[atoms_id_[id_s]] = sas_atom.atomArea(id_s)
                sas = sas.reshape(-1, 37)
                sas[torch.isnan(sas).bool()] = 0
            
            atom_pos_all37 = atom_pos_all.clone()
            atom_masks_all37 = atom_masks_all.clone()
            # only save alpahc 
            atom_pos = atom_pos_all[:, 1]
            residue_type = residue_type_all[:, 1]

            # get inter atoms
            res_mask = torch.zeros(atom_type_all.shape).bool()
            res_mask[:, 1] = True
            res_mask = res_mask.reshape(-1)

            inter_mask = atom_masks_all.clone().reshape(-1)

            atom_feat = atom_feat.reshape(-1, 5)[inter_mask]
            atom_pos_all_origin = atom_pos_all_origin.reshape(-1, 3)[inter_mask]
            atom_pos_all = atom_pos_all.reshape(-1, 3)[inter_mask]
            atom_type_all = atom_type_all.reshape(-1)[inter_mask]
            atom_type_origin = atom_type_origin.reshape(-1)[inter_mask]
            residue_pos_all = residue_pos_all.reshape(-1)[inter_mask]
            residue_type_all = residue_type_all.reshape(-1)[inter_mask]
            pocket_label_all = pocket_label_all.reshape(-1)[inter_mask]
            res_mask = res_mask[inter_mask]
            if self.args.use_sas:
                sas_res = sas[:, 1]
                sas_atom = sas.reshape(-1)[inter_mask]
                if self.args.use_virtual_node:
                    sas_res = torch.cat([sas_res, torch.zeros(1)], dim=0)
                    sas_atom = torch.cat([sas_atom, torch.zeros(1)], dim=0)
            n_atom_all = atom_pos_all.shape[0]

            ###################################################
            distance_mat = distance_matrix(atom_pos, atom_pos)
            knn_res = min(self.args.knn_res, distance_mat.shape[0] - 1)

            D_adjust = torch.from_numpy(distance_mat)
            _, e = torch.topk(D_adjust, min(knn_res, D_adjust.shape[-1]), dim=-1, largest=False)
            
            cols = e[:, :knn_res].reshape(-1)
            rows = np.arange(distance_mat.shape[0]).reshape(distance_mat.shape[0], 1)
            rows = np.tile(rows, (1, knn_res)).reshape(-1)
            assert cols.shape == rows.shape, (cols.shape, rows.shape)
            assert cols.shape[0] > 0, idx

            ###################################################
            distance_mat_aa = distance_matrix(atom_pos_all, atom_pos_all)
            knn_atom = min(self.args.knn_atom2atom, distance_mat_aa.shape[1] - 1)
            assert knn_atom > 1
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
                atom_pos_all_origin = torch.from_numpy(np.concatenate([atom_pos_all_origin, np.zeros((1, 3))], axis=0))
                residue_type_all = np.concatenate([residue_type_all, np.ones(1) * 23], axis=0)
                atom_type_all = np.concatenate([atom_type_all, np.ones(1) * 38], axis=0)
                atom_pos_pred_index = torch.from_numpy(np.concatenate([atom_type_origin, np.zeros(1)], axis=0))
                atom_type_origin = torch.from_numpy(np.concatenate([atom_type_origin, np.ones(1) * 38], axis=0))
                pocket_label_all = torch.cat([pocket_label_all, torch.zeros(1)])
                atom_feat = torch.cat([atom_feat, torch.zeros(1, 5)])
                if ss is not None:
                    ss = torch.cat([ss, torch.zeros(1)]).long()
                if gb_feat is not None:
                    gb_feat = torch.cat([gb_feat, torch.zeros((1, gb_feat.shape[1]))])

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
                residue_type = torch.cat([residue_type, torch.ones(1).long() * 23])
                residue_pos_all = torch.cat([residue_pos_all, torch.ones(1).long() * num_real])
                residue_type_origin = torch.cat([residue_type_origin, torch.ones(1).long() * 23])
                atom_pos_origin = torch.cat([atom_pos_origin, torch.zeros(1, 3)], dim=0)
                pocket_label = torch.cat([pocket_label, torch.zeros((1, 37)).long()])
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
            assert torch.all(atom_type_origin != 37)
            assert torch.all(residue_type_origin != 22)

            atom_label = (pocket_label > 0).long()
            label = (torch.sum(atom_label > 0, axis=1) > 0).long()
            if self.args.use_virtual_node:
                esm_feat = torch.cat([esm_feat, esm_cls_feat.unsqueeze(0)])
                assert esm_feat.shape[0] == atom_pos.shape[0]
            is_train = torch.Tensor([self.is_train * 1])

            if self.args.edge_mask_prob > 0 and self.is_train:
                select_num = int((1 - self.args.edge_mask_prob) * edge_index.shape[0])
                index_select = np.random.choice(np.arange(edge_index.shape[0]), select_num, replace=False)
                edge_index = edge_index[index_select]
                edge_vec = edge_vec[index_select]

                select_num = int((1 - self.args.edge_mask_prob) * aa_edge_index.shape[0])
                index_select = np.random.choice(np.arange(aa_edge_index.shape[0]), select_num, replace=False)
                aa_edge_index = aa_edge_index[index_select]
                edge_aa_vec = edge_aa_vec[index_select]

            return {
                "atom_pos": atom_pos_all,
                "residue_pos": (residue_pos.long(), atom_pos.shape[0]),
                "residue_pos_all": (residue_pos_all.long(), atom_pos.shape[0]),
                "atom_type": torch.from_numpy(atom_type_all).long(),
                "residue_type": torch.from_numpy(residue_type_all).long(),
                "edge_index": (edge_index.long(), atom_pos.shape[0]),
                "atom_pos_origin": atom_pos_origin,
                "aa_edge_index": (aa_edge_index.long(), atom_type_all.shape[0]),
                "atom_pos_all_origin": atom_pos_all_origin,
                "residue_type_origin": residue_type_origin,
                "esm_feat": esm_feat,
                "label": label,
                "pocket_label_all": pocket_label_all,
                "atom_type_origin": atom_type_origin,
                "batch_index": torch.ones(atom_pos_all.shape[0]),
                "batch_index_res": torch.ones(atom_pos.shape[0]),
                "atom_pos": atom_pos_all,
                "aa_edge_index": (aa_edge_index.long(), atom_type_all.shape[0]),
                "atom_feat": atom_feat,
                "edge_vec": edge_vec,
                "edge_aa_vec": edge_aa_vec,
                "res_mask": res_mask.bool(),
                "idx": torch.Tensor([idx]),
                "atom_sas": sas_atom if self.args.use_sas else None,
                "gb_feat": gb_feat,
                "ss": ss,
            }



class PocketTaskDataset(BaseWrapperDataset):
    def __init__(self, dataset, key="pocket_label"):
        self.dataset = dataset
        self.set_epoch(None)
        self.key = key

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx: int):
        return self.__getitem_cached__(self.epoch, idx)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch, index):
        return self.dataset[index][
            self.key
        ]#.long()

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        # return pad_1d(samples)
        return torch.cat(samples, dim=0)
    
class ListDataset(BaseWrapperDataset):
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
    def collater(self, samples):
        # return pad_1d(samples)
        return samples
    
class DrugProteinDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        key="protein",
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
    def collater(self, samples):
        return samples[0].pack(samples)