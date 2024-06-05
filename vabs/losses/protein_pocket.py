# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import scipy.stats as stats
import numpy as np
import pickle

from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.utils import one_hot
from sklearn.metrics import roc_auc_score
import vabs.losses.rc as rc
def masked_mean(mask, value, dim, eps=1e-10, keepdim=False):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim, keepdim=keepdim) / (
        eps + torch.sum(mask, dim=dim, keepdim=keepdim)
    )


rc_chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # UNK
]


@register_loss("protein_pretrain")
class ProteinLoss(UnicoreLoss):
    f_thresh = 0.03
    e_thresh = 0.02

    def __init__(self, task):
        super().__init__(task)
        self.pocket_loss_weight = 1
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, model, sample, reduce=True):
        res_pred_type, res_pred_type_halfway, res_pred_type_index, pocket_pred, atom_pred_pos, atom_pred_pos_index, dihedral_perd, _, sas_pred, num_updates = model(
            **sample["net_input"],
        )

        bsz = sample["targets"]["batch_index"][-1].data + 1
        res_cnt = res_pred_type_index.sum()
        if self.args.res_type_pred:
            res_type_loss = F.cross_entropy(res_pred_type.float().view(-1, res_pred_type.shape[-1]), sample["targets"]['residue_type'][res_pred_type_index.bool()].view(-1).long(), reduction="mean")
            res_pred_type_t = self.softmax(res_pred_type.detach())
            res_type_acc = (torch.argmax(res_pred_type_t, axis=-1) == sample["targets"]['residue_type'][res_pred_type_index.bool()]).sum() / res_pred_type_index.sum()
            if self.args.refine:
                res_pred_loss_halfway = F.cross_entropy(res_pred_type_halfway.float().view(-1, res_pred_type.shape[-1]), sample["targets"]['residue_type'][res_pred_type_index.bool()].view(-1).long(), reduction="mean")
            else:
                res_pred_loss_halfway = torch.Tensor([0]).to(res_pred_type_index.device)
        else:
            res_type_loss = torch.Tensor([0]).to(res_pred_type_index.device)
            res_type_acc = torch.Tensor([0]).to(res_pred_type_index.device)

        if self.args.pos_pred:
            def get_pos_loss(pos_pred, pos_target, pos_mask):
                pos_mask = pos_mask.unsqueeze(-1)
                pos_pred = pos_pred.float() * pos_mask
                pos_loss = torch.nn.L1Loss(reduction="none")(
                    pos_pred,
                    pos_target * pos_mask,
                ).sum(dim=(-1, -2))
                pos_cnt = pos_mask.squeeze(-1).sum(dim=-1) + 1e-10
                pos_loss = (pos_loss / pos_cnt).mean()
                return pos_loss

            atom_pos_loss = get_pos_loss(atom_pred_pos, sample["targets"]["atom_pos"], atom_pred_pos_index)

            def get_dist_loss(dist_pred, dist_target, pair_mask, return_sum=True):
                dist_cnt = pair_mask.sum(dim=(-1, -2)) + 1e-10
                dist_pred = dist_pred.float() * pair_mask
                dist_loss = torch.nn.L1Loss(reduction="none")(
                    dist_pred,
                    dist_target,
                ).sum(dim=(-1, -2))
                if return_sum:
                    return (dist_loss / dist_cnt).mean()
                else:
                    return dist_loss / dist_cnt

            atom_pair_mask = ((sample["targets"]["batch_index"].unsqueeze(-1) - sample["targets"]["batch_index"].unsqueeze(-2).float()) == 0).long()
            atom_pair_mask = atom_pair_mask * ((atom_pred_pos_index.unsqueeze(-1) + atom_pred_pos_index.unsqueeze(-2)) > 0).long()

            atom_dist_target = (sample["targets"]["atom_pos"].unsqueeze(-2) - sample["targets"]["atom_pos"].unsqueeze(-3)).norm(dim=-1)
            atom_dist_target = atom_dist_target * atom_pair_mask
            atom_dist_pred = (atom_pred_pos.unsqueeze(-2) - atom_pred_pos.unsqueeze(-3)).norm(dim=-1)

            atom_dist_loss = get_dist_loss(atom_dist_pred, atom_dist_target, atom_pair_mask)
        else:
            atom_pos_loss = torch.Tensor([0]).to(res_type_loss.device)
            atom_dist_loss = torch.Tensor([0]).to(res_type_loss.device)
        res_mask = sample["net_input"]["res_mask"]

        if self.args.dihedral_pred:
            def supervised_chi_loss(
                pred_angles_sin_cos: torch.Tensor,
                pred_unnormed_angles_sin_cos: torch.Tensor,
                true_angles_sin_cos: torch.Tensor,
                aatype: torch.Tensor,
                seq_mask: torch.Tensor,
                chi_mask: torch.Tensor,
                angle_norm_weight: float=1,
                chi_weight: float=1,
                eps=1e-6,
                **kwargs,
            ) -> torch.Tensor:
                # TODO: refactor this.
                pred_angles_sin_cos = pred_angles_sin_cos.float().unsqueeze(0)
                pred_unnormed_angles_sin_cos = pred_unnormed_angles_sin_cos.float().unsqueeze(0)
                true_angles_sin_cos = true_angles_sin_cos.unsqueeze(0).float()
                seq_mask = seq_mask.float()
                chi_mask = chi_mask.float().unsqueeze(0)
                aatype = aatype.unsqueeze(0)
                # pred_angles = pred_angles_sin_cos[..., 3:, :]
                pred_angles = pred_angles_sin_cos

                greater = aatype >= 21
                aatype[greater] = 20
                residue_type_one_hot = one_hot(
                    aatype,
                    rc.restype_num + 1,
                )
                chi_pi_periodic = torch.einsum(
                    "ijk, kl->ijl",
                    residue_type_one_hot.type(pred_angles_sin_cos.dtype),
                    pred_angles_sin_cos.new_tensor(rc_chi_pi_periodic),
                )
                # true_chi = true_angles_sin_cos[..., 3:, :]
                true_chi = true_angles_sin_cos
                shifted_mask = (1.0 - 2.0 * chi_pi_periodic)[None, ..., None]
                true_chi_shifted = shifted_mask * true_chi
                sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
                sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_angles) ** 2, dim=-1)
                sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
                # permute nblock and batch dim
                sq_chi_error = sq_chi_error.transpose(0, 1)
                chi_mask = chi_mask.reshape(1, -1, 7, 2).sum(-1).bool()
                mask = chi_mask.unsqueeze(1)
                sq_chi_loss = masked_mean(mask, sq_chi_error, dim=(-1, -2, -3))
                dih_loss = chi_weight * sq_chi_loss

                angle_norm = torch.sqrt(torch.sum(pred_unnormed_angles_sin_cos**2, dim=-1) + eps)
                norm_error = torch.abs(angle_norm - 1.0)
                mask = seq_mask[..., None, :, None]
                angle_norm_loss = masked_mean(mask, norm_error, dim=(-1, -2, -3))

                dih_loss = dih_loss + angle_norm_weight * angle_norm_loss
                return dih_loss
            unnormalized_dihedral_perd, dihedral_perd, res_pred_type_index_t = dihedral_perd
            dihedral_loss = supervised_chi_loss(pred_angles_sin_cos=dihedral_perd, 
                                                pred_unnormed_angles_sin_cos=unnormalized_dihedral_perd, 
                                                true_angles_sin_cos=sample["targets"]["torsion"].reshape(-1, 7, 2), 
                                                aatype=sample["targets"]['residue_type'][res_pred_type_index_t][res_mask[res_pred_type_index_t]],
                                                seq_mask=torch.ones(sample["targets"]['residue_type'][res_pred_type_index_t][res_mask[res_pred_type_index_t]].shape[0]).to(sample["targets"]["torsion_mask"].device),
                                                chi_mask=sample["targets"]["torsion_mask"])
        else:
            dihedral_loss = torch.Tensor([0]).to(res_type_loss.device)

        if self.args.sas_pred:
            sas_loss = torch.nn.L1Loss(reduction="none")(
                    sas_pred.squeeze(-1),
                    sample["targets"]["atom_sas"],
                ).mean()
        else:
            sas_loss = torch.Tensor([0]).to(res_type_loss.device)

        loss = res_type_loss + res_pred_loss_halfway + atom_dist_loss + atom_pos_loss + dihedral_loss + sas_loss
        logging_output = {
            "res_type_loss": res_type_loss.data,
            "dihedral_loss": dihedral_loss.data,
            "atom_pos_loss": atom_pos_loss.data,
            "atom_dist_loss": atom_dist_loss.data,
            "sas_loss": sas_loss.data,
            "loss": loss.data,
            "sample_size": 1,
            "bsz": bsz,
            "res_type_acc": res_type_acc.data,
            "res_cnt": res_cnt,
        }
        # assert 1==0, logging_output
        return loss, 1, logging_output
        
    @staticmethod
    def reduce_metrics(logging_outputs, split) -> None:
        """Aggregate logging outputs from data parallel training."""
        res_type_loss = sum(log.get("res_type_loss", 0) for log in logging_outputs)
        dihedral_loss = sum(log.get("dihedral_loss", 0) for log in logging_outputs)
        atom_pos_loss = sum(log.get("atom_pos_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        res_type_acc = sum(log.get("res_type_acc", 0) for log in logging_outputs)
        atom_dist_loss = sum(log.get("atom_dist_loss", 0) for log in logging_outputs)
        sas_loss = sum(log.get("sas_loss", 0) for log in logging_outputs)
        loss = sum(log.get("loss", 0) for log in logging_outputs)

        res_cnt = sum(log.get("res_cnt", 0) for log in logging_outputs)
        
        metrics.log_scalar("loss", loss/sample_size, sample_size, round=6)
        metrics.log_scalar("sas_loss", sas_loss/sample_size, sample_size, round=6)
        metrics.log_scalar("atom_pos_loss", atom_pos_loss/sample_size, sample_size, round=6)
        metrics.log_scalar("dihedral_loss", dihedral_loss/sample_size, sample_size, round=6)
        metrics.log_scalar("atom_dist_loss", atom_dist_loss/sample_size, sample_size, round=6)
        metrics.log_scalar("res_type_loss", res_type_loss/sample_size, sample_size, round=6)
        metrics.log_scalar("res_type_acc", res_type_acc/sample_size, sample_size, round=6)
        metrics.log_scalar("res_cnt", res_cnt/sample_size, sample_size, round=6)
        
        
        

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train