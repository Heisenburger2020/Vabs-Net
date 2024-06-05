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

def calculate_ce(preds, targets):
    targets = targets.unsqueeze(-1)
    preds = F.log_softmax(preds, dim=-1, dtype=torch.float32)
    targets = torch.cat((1-targets, targets), axis=-1)
    loss = -1 * (torch.sum(preds[:,0] * targets[:,0]) + torch.sum(preds[:,1] * targets[:,1]) )
    return loss


def soft_iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()  
    union = pred.sum() + target.sum() - intersection.sum()
    iou = (intersection + 1e-7) / (union + 1e-7)
    soft_iou_loss = 1 - iou.mean()

    return soft_iou_loss

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


@register_loss("protein_pocket_ft")
class ProteinftLoss(UnicoreLoss):
    f_thresh = 0.03
    e_thresh = 0.02

    def __init__(self, task):
        super().__init__(task)
        self.pocket_loss_weight = 1
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, model, sample, reduce=True):
        
        res_type_pred, res_pred_type_halfway, res_pred_type_index, pred_pocket_all, atom_pos_pred, atom_pred_pos_index, dihedral_perd, pred_label, _, num_updates = model(
            **sample["net_input"],
        )

        assert not torch.isnan(pred_pocket_all).any().item()
        pred_pocket = self.softmax(pred_pocket_all.detach())
        # pred_pocket = torch_scatter.scatter(pred_pocket.cpu(), sample["net_input"]["residue_idx_all"].long().cpu(), dim=0, dim_size=sample["net_input"]["res_mask"].sum(), reduce="max")
        pred_pocket = pred_pocket[sample["net_input"]["res_mask"]]
        bsz = sample["targets"]["batch_index"][-1].data + 1

        assert pred_pocket_all.shape[0] == sample["targets"]['pocket_label_all'].shape[0], (pred_pocket_all.shape[0], sample["targets"]['pocket_label_all'].shape)
        assert pred_pocket.shape[0] == sample["targets"]['label'].shape[0], (pred_pocket.shape, sample["targets"]['label'].shape)
        pocket_loss = F.nll_loss(torch.log(pred_pocket.float().view(-1, pred_pocket.shape[-1])).to(pred_pocket_all.device), sample["targets"]['label'].view(-1).long(), reduction="mean")
        res_cnt = sample["targets"]['label'].view(-1).size(0)
        pocket_loss_all = F.cross_entropy(pred_pocket_all.float().view(-1, pred_pocket_all.shape[-1]), sample["targets"]['pocket_label_all'].view(-1).long(), reduction="mean")
        pred_pocket_t = pred_pocket.detach().view(-1)
        pocket_acc = (torch.argmax(pred_pocket_t, axis=-1) == sample["targets"]['label']).sum() / res_cnt

        batch_index = sample["targets"]["batch_index"].view(-1)
        batch_index_res = sample["targets"]["batch_index_res"].view(-1)
        targets = sample["targets"]['label'].view(-1)
        targets_all = sample["targets"]['pocket_label_all'].view(-1)
        auc_sum = 0
        auc_vote_sum = 0
        iou_sum = 0
        iou_count = 0
        iou_vote_sum = 0
        iou_vote_count = 0
        auc_count = 0
        auc_vote_count = 0
        pocket_count = 0
        mcc_all = 0
        f1_all, recall_all, precision_all, accuracy_all = 0, 0, 0, 0

        bsz_count = int(bsz.cpu().data)
        pred_pocket_all = self.softmax(pred_pocket_all.detach())

        for i in range(bsz_count):
            assert len(targets_all[batch_index==i]) > 0
            label_true = (targets_all[batch_index==i].detach().cpu().numpy() > 0).astype(np.int)
            label_score = pred_pocket_all[batch_index==i][:,1].detach().cpu().numpy()
            batch_index = batch_index.cpu()
            batch_index_res = batch_index_res.cpu()
            try:
                auc_sum += roc_auc_score(label_true, label_score)
                auc_count += 1  
            except:
                pass
            label_score = (label_score > 0.5).astype(np.int)
            union = ((label_true + label_score) > 0).sum()
            intersection = (label_true * label_score > 0).sum()
            if union > 0:
                iou = (intersection / union).item() 
                iou_sum += iou
                iou_count += 1
            label_true = (targets[batch_index_res==i].detach().cpu().numpy() > 0).astype(np.int)
            label_vote = np.array(pred_pocket[batch_index_res==i][:, 1].detach().cpu())
            assert np.all(label_vote <= 1) and np.all(label_vote >= 0), label_vote
            assert np.all(label_score <= 1) and np.all(label_score >= 0), label_score
            pocket_count += label_true.sum()

            try:
                auc_vote_sum += roc_auc_score(label_true, label_vote)
                auc_vote_count += 1  
            except:
                pass
            assert auc_count == auc_vote_count


            label_vote = (label_vote > 0.5).astype(np.int)
            union = ((label_true + label_vote) > 0).sum()
            intersection = (label_true * label_vote > 0).sum()
            def calculate_f1_recall_precision(predictions, targets):
                # 将预测值和真实标签转换为0/1的张量
                predictions = torch.from_numpy(predictions).float()
                targets = torch.from_numpy(targets).float()
                # 计算TP、FP和FN的数量
                TP = torch.sum(predictions * targets)
                TN = torch.sum((1 - predictions) * (1 - targets))
                FP = torch.sum(predictions * (1 - targets))
                FN = torch.sum((1 - predictions) * targets)
                precision = TP / (TP + FP + 1e-8)  # 避免除零错误，加入极小值epsilon
                recall = TP / (TP + FN + 1e-8)
                mcc = (TP * TN - FP * FN) / (torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-5)

                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                accuracy = (TP + torch.sum(1 - targets)) / targets.size(0)
                return f1, recall, precision, accuracy, mcc
            f1, recall, precision, accuracy, mcc = calculate_f1_recall_precision(label_vote, label_true)
            f1_all += f1
            recall_all += recall
            precision_all += precision
            accuracy_all += accuracy
            mcc_all += mcc
            if union > 0:
                iou = (intersection / union).item() 
                iou_vote_sum += iou

        loss = pocket_loss_all
        
        logging_output = {
            "loss": loss.data,
            "sample_size": 1,
            "bsz": bsz,
            "res_cnt": res_cnt/bsz,
            "pocket_loss": pocket_loss.data,
            "pocket_acc": pocket_acc,
            "auc": auc_sum,
            "auc_vote": auc_vote_sum,
            "iou": iou_sum,
            "iou_vote": iou_vote_sum,
            "auc_count": auc_count,
            "iou_count": iou_count,
            "mcc": mcc_all / bsz,
            "pocket_loss_all": pocket_loss_all.data,
            "f1": f1_all/bsz,
            "recall": recall_all/bsz,
            "precision": precision_all/bsz,
            "accuracy": accuracy_all/bsz,
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

        auc = sum(log.get("auc", 0) for log in logging_outputs)
        auc_vote = sum(log.get("auc_vote", 0) for log in logging_outputs)
        iou_vote = sum(log.get("iou_vote", 0) for log in logging_outputs)
        iou = sum(log.get("iou", 0) for log in logging_outputs)

        auc_count = sum(log.get("auc_count", 0) for log in logging_outputs)
        iou_count = sum(log.get("iou_count", 0) for log in logging_outputs)
        mcc = sum(log.get("mcc", 0) for log in logging_outputs)
        f1 = sum(log.get("f1", 0) for log in logging_outputs)
        recall = sum(log.get("recall", 0) for log in logging_outputs)
        precision = sum(log.get("precision", 0) for log in logging_outputs)
        accuracy = sum(log.get("accuracy", 0) for log in logging_outputs)
        metrics.log_scalar("f1", f1/sample_size, sample_size, round=6)
        metrics.log_scalar("recall", recall/sample_size, sample_size, round=6)
        metrics.log_scalar("precision", precision/sample_size, sample_size, round=6)
        metrics.log_scalar("accuracy", accuracy/sample_size, sample_size, round=6)

        if mcc > 0:
            metrics.log_scalar("mcc", mcc/sample_size, sample_size, round=6)

        if loss > 0:
            metrics.log_scalar("loss", loss/sample_size, sample_size, round=6)

        if res_cnt > 0:
            metrics.log_scalar("res_cnt", res_cnt/sample_size, sample_size, round=6)
        
        if pocket_loss > 0:
            metrics.log_scalar("pocket_loss", pocket_loss/sample_size, sample_size, round=6)
        
        if auc_count > 0:
            metrics.log_scalar("auc", auc/auc_count, auc_count, round=6)
            metrics.log_scalar("auc_vote", auc_vote/auc_count, auc_count, round=6)
        if iou_count > 0:
            metrics.log_scalar("iou", iou/iou_count, iou_count, round=6)
            metrics.log_scalar("iou_vote", iou_vote/iou_count, iou_count, round=6)
        
        

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train