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
from vabs.losses.rc import *
from torch_scatter import scatter_mean

def f1_max(pred, target):
    """
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

@register_loss("protein_pocket_ec")
class ECProteinftLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.pocket_loss_weight = 1
        self.softmax = nn.Softmax(dim=-1)
        if self.args.use_asl:
            self.loss = AsymmetricLossOptimized(gamma_neg=self.args.gamma_neg)
        else:
            self.loss = RankLoss()


    def forward(self, model, sample, reduce=True):
        
        res_type_pred, res_pred_type_halfway, res_pred_type_index, _, atom_pos_pred, atom_pred_pos_index, dihedral_perd, pred_label, _, num_updates = model(
            **sample["net_input"],
        )

        assert not torch.isnan(pred_label).any().item()
        bsz = sample["targets"]["batch_index"][-1].data + 1
        
        pocket_loss = self.loss(pred_label, sample["targets"]['label'].type(pred_label.dtype))
        res_cnt = sample["targets"]['label'].view(-1).size(0)

        cal_f1 = 0
        try:
            idx = sample["net_input"]["idx"]
            if sample["net_input"]["is_train"][0] > 0:
                cal_f1 = 1
                for i in range(int(bsz)):
                    if sample["net_input"]["is_train"][i] == 0:
                        continue
                    with open(os.path.join(self.args.spearman, f'{int(idx[i])}.pkl'), 'wb') as f:
                        pickle.dump((torch.sigmoid(pred_label.detach()).cpu(), sample["targets"]['label'].long().cpu()), f)
        except:
            print(">>>>>>>>> warning >>>>>>>>")

        loss = pocket_loss
        f1_bsz = f1_max(torch.sigmoid(pred_label.detach()).cpu(), sample["targets"]['label'].long().cpu())
        
        logging_output = {
            "loss": loss.data,
            "sample_size": 1,
            "bsz": bsz,
            "res_cnt": res_cnt/bsz,
            "pocket_loss": pocket_loss.data,
            "f1": f1_bsz,
            "cal_f1": cal_f1,
        }
        # assert 1==0, logging_output
        return loss, 1, logging_output
        
    @staticmethod
    def reduce_metrics(self, logging_outputs, split) -> None:
        """Aggregate logging outputs from data parallel training."""
        # split = self.split
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        loss = sum(log.get("loss", 0) for log in logging_outputs)

        res_cnt = sum(log.get("res_cnt", 0) for log in logging_outputs)

        cal_f1 = sum(log.get("cal_f1", 0) for log in logging_outputs)
        cal_f1 = cal_f1 * self.args.batch_size_valid
        if cal_f1 > 1500:
            print("num_sample", cal_f1)
            all_pred = []
            all_target = []
            if self.args.cls_type != "EC":
                if cal_f1 < 3400:
                    start = 0
                    end = 3322
                else:
                    start = 10000
                    end = 10000 + 3415
            else:
                if cal_f1 < 1900:
                    start = 0
                    end = 1729
                else:
                    start = 10000
                    end = 10000 + 1919

            for i in range(start, end):
                with open(os.path.join(self.args.spearman, f'{int(i)}.pkl'), 'rb') as f:
                    loaded_tensor = pickle.load(f)
                    all_pred.append(loaded_tensor[0])
                    all_target.append(loaded_tensor[1])
            all_pred = torch.cat(all_pred, dim=0).float()
            all_target = torch.cat(all_target, dim=0).float()
            assert not torch.isnan(all_pred).any().item()
            assert len(all_pred.shape) == 2, all_pred.shape
            f1 = f1_max(all_pred, all_target)
            metrics.log_scalar("f1", f1, 1, round=6)
        else:
            pass
        f1 = sum(log.get("f1", 0) for log in logging_outputs)
        metrics.log_scalar("f1_bsz", f1/sample_size, sample_size, round=6)

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
