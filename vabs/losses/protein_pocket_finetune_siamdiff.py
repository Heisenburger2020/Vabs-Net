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
import torch_scatter
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

from torch_scatter import scatter_mean

def f1_max(pred, target):
    """
    From torchdrug:
    
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()

@register_loss("protein_ec_siamdiff")
class siamdiffECProteinftLoss(UnicoreLoss):
    f_thresh = 0.03
    e_thresh = 0.02

    def __init__(self, task):
        super().__init__(task)
        self.pocket_loss_weight = 1
        self.softmax = nn.Softmax(dim=-1)
        self.BCELoss = torch.nn.BCEWithLogitsLoss()

    def forward(self, model, sample, reduce=True):
        
        pred_label, num_updates = model(
            **sample["net_input"],
        )

        assert not torch.isnan(pred_label).any().item()
        # pred_pocket = self.softmax(pred_label.detach())
        # pred_pocket = torch_scatter.scatter(pred_pocket.cpu(), sample["net_input"]["residue_idx_all"].long().cpu(), dim=0, dim_size=sample["net_input"]["res_mask"].sum(), reduce="max")
        bsz = sample["targets"]["batch_index"][-1].data + 1
        

        pocket_loss = self.BCELoss(pred_label, sample["targets"]['pocket_label_all'].type(pred_label.dtype))

        res_cnt = sample["targets"]['pocket_label_all'].view(-1).size(0)


        f1_all = f1_max(self.softmax(pred_label.detach()), sample["targets"]['pocket_label_all'].long()).mean()

        loss = pocket_loss
        
        logging_output = {
            "loss": loss.data,
            "sample_size": 1,
            "bsz": bsz,
            "res_cnt": res_cnt/bsz,
            "pocket_loss": pocket_loss.data,
            "f1": f1_all,
        }
        # assert 1==0, logging_output
        return loss, 1, logging_output
        
    @staticmethod
    def reduce_metrics( logging_outputs, splits) -> None:
        """Aggregate logging outputs from data parallel training."""
        print(logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        loss = sum(log.get("loss", 0) for log in logging_outputs)

        res_cnt = sum(log.get("res_cnt", 0) for log in logging_outputs)

        f1 = sum(log.get("f1", 0) for log in logging_outputs)
        metrics.log_scalar("f1", f1/sample_size, sample_size, round=6)

        metrics.log_scalar("loss", loss/sample_size, sample_size, round=6)

        if res_cnt > 0:
            metrics.log_scalar("res_cnt", res_cnt/sample_size, sample_size, round=6)
        

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train