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

@register_loss("protein_pocket_ft_esm")
class ESMProteinftLoss(UnicoreLoss):
    f_thresh = 0.03
    e_thresh = 0.02

    def __init__(self, task):
        super().__init__(task)
        self.pocket_loss_weight = 1
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, model, sample, reduce=True):
        
        pred_pocket_all, num_updates = model(
            **sample["net_input"],
        )

        assert not torch.isnan(pred_pocket_all).any().item()
        pred_pocket = self.softmax(pred_pocket_all.detach())
        bsz = sample["targets"]["batch_index"][-1].data + 1

        pocket_loss = F.cross_entropy(pred_pocket_all.view(-1, pred_pocket.shape[-1]).to(pred_pocket_all.device), sample["targets"]['pocket_label_all'].view(-1).long(), reduction="mean")
        res_cnt = sample["targets"]['pocket_label_all'].view(-1).size(0)

        batch_index = sample["targets"]["batch_index"].view(-1)
        targets_all = sample["targets"]['pocket_label_all'].view(-1)
        auc_vote_sum = 0
        iou_count = 0
        iou_vote_sum = 0
        auc_count = 0
        auc_vote_count = 0
        pocket_count = 0

        bsz_count = int(bsz.cpu().data)

        for i in range(bsz_count):
            assert len(targets_all[batch_index==i]) > 0
            label_true = (targets_all[batch_index==i].detach().cpu().numpy() > 0).astype(np.int)
            label_score = pred_pocket[batch_index==i][:,1].detach().cpu().numpy()
            pocket_count += label_true.sum()

            try:
                auc_vote_sum += roc_auc_score(label_true, label_score)
                auc_vote_count += 1 
            except:
                pass
            
            label_vote = (label_score > 0.5).astype(np.int)
            union = ((label_true + label_vote) > 0).sum()
            intersection = (label_true * label_vote > 0).sum()
            if union > 0:
                iou = (intersection / union).item() 
                iou_vote_sum += iou
                iou_count += 1

        loss = pocket_loss
        
        logging_output = {
            "loss": loss.data,
            "sample_size": 1,
            "bsz": bsz,
            "res_cnt": res_cnt/bsz,
            "pocket_loss": pocket_loss.data,
            "auc_vote": auc_vote_sum,
            "iou_vote": iou_vote_sum,
            "auc_count": auc_vote_count,
            "iou_count": iou_count,
            "pocket_loss_all": pocket_loss.data,
        }
        # assert 1==0, logging_output
        return loss, 1, logging_output
        
    @staticmethod
    def reduce_metrics(logging_outputs, split) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        loss = sum(log.get("loss", 0) for log in logging_outputs)

        res_cnt = sum(log.get("res_cnt", 0) for log in logging_outputs)

        pocket_loss = sum(log.get("pocket_loss", 0) for log in logging_outputs)

        auc_vote = sum(log.get("auc_vote", 0) for log in logging_outputs)
        iou_vote = sum(log.get("iou_vote", 0) for log in logging_outputs)

        auc_count = sum(log.get("auc_count", 0) for log in logging_outputs)
        iou_count = sum(log.get("iou_count", 0) for log in logging_outputs)

        if loss > 0:
            metrics.log_scalar("loss", loss/sample_size, sample_size, round=6)

        if res_cnt > 0:
            metrics.log_scalar("res_cnt", res_cnt/sample_size, sample_size, round=6)
        if pocket_loss > 0:
            metrics.log_scalar("pocket_loss", pocket_loss/sample_size, sample_size, round=6)
        if auc_count > 0:
            metrics.log_scalar("auc_vote", auc_vote/auc_count, auc_count, round=6)
        if iou_count > 0:
            metrics.log_scalar("iou_vote", iou_vote/iou_count, iou_count, round=6)
        
        

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
    

@register_loss("protein_pocket_ft_siam")
class siamdiffProteinftLoss(UnicoreLoss):
    f_thresh = 0.03
    e_thresh = 0.02

    def __init__(self, task):
        super().__init__(task)
        self.pocket_loss_weight = 1
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, model, sample, reduce=True):
        
        pred_pocket_all, num_updates = model(
            **sample["net_input"],
        )

        assert not torch.isnan(pred_pocket_all).any().item()
        pred_pocket = self.softmax(pred_pocket_all.detach())
        bsz = sample["targets"]["batch_index"][-1].data + 1

        pocket_loss = F.cross_entropy(pred_pocket_all.view(-1, pred_pocket.shape[-1]).to(pred_pocket_all.device), sample["targets"]['pocket_label_all'].view(-1).long(), reduction="mean")
        res_cnt = sample["targets"]['pocket_label_all'].view(-1).size(0)

        res_mask = sample["targets"]["res_mask"].view(-1)
        batch_index = sample["targets"]["batch_index"].view(-1)
        targets_all = sample["targets"]['pocket_label_all'].view(-1)
        auc_vote_sum = 0
        iou_count = 0
        iou_vote_sum = 0
        auc_count = 0
        auc_vote_count = 0
        pocket_count = 0

        bsz_count = int(bsz.cpu().data)

        for i in range(bsz_count):
            assert len(targets_all[batch_index==i]) > 0
            res_mask_ = res_mask[batch_index==i]
            label_true = (targets_all[batch_index==i].detach().cpu().numpy() > 0).astype(np.int)[res_mask_.cpu()]
            label_score = pred_pocket[batch_index==i][:,1].detach().cpu().numpy()[res_mask_.cpu()]
            pocket_count += label_true.sum()

            try:
                auc_vote_sum += roc_auc_score(label_true, label_score)
                auc_vote_count += 1 
            except:
                pass
            
            label_vote = (label_score > 0.5).astype(np.int)
            union = ((label_true + label_vote) > 0).sum()
            intersection = (label_true * label_vote > 0).sum()
            if union > 0:
                iou = (intersection / union).item() 
                iou_vote_sum += iou
                iou_count += 1

        loss = pocket_loss
        
        logging_output = {
            "loss": loss.data,
            "sample_size": 1,
            "bsz": bsz,
            "res_cnt": res_cnt/bsz,
            "pocket_loss": pocket_loss.data,
            "auc_vote": auc_vote_sum,
            "iou_vote": iou_vote_sum,
            "auc_count": auc_vote_count,
            "iou_count": iou_count,
            "pocket_loss_all": pocket_loss.data,
        }
        # assert 1==0, logging_output
        return loss, 1, logging_output
        
    @staticmethod
    def reduce_metrics(logging_outputs, split) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        loss = sum(log.get("loss", 0) for log in logging_outputs)

        res_cnt = sum(log.get("res_cnt", 0) for log in logging_outputs)

        pocket_loss = sum(log.get("pocket_loss", 0) for log in logging_outputs)

        auc_vote = sum(log.get("auc_vote", 0) for log in logging_outputs)
        iou_vote = sum(log.get("iou_vote", 0) for log in logging_outputs)

        auc_count = sum(log.get("auc_count", 0) for log in logging_outputs)
        iou_count = sum(log.get("iou_count", 0) for log in logging_outputs)

        if loss > 0:
            metrics.log_scalar("loss", loss/sample_size, sample_size, round=6)

        if res_cnt > 0:
            metrics.log_scalar("res_cnt", res_cnt/sample_size, sample_size, round=6)
        if pocket_loss > 0:
            metrics.log_scalar("pocket_loss", pocket_loss/sample_size, sample_size, round=6)
        if auc_count > 0:
            metrics.log_scalar("auc_vote", auc_vote/auc_count, auc_count, round=6)
        if iou_count > 0:
            metrics.log_scalar("iou_vote", iou_vote/iou_count, iou_count, round=6)
        
        

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train