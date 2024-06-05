# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#


import os
import pickle
import torch
import math
import numpy as np

from unicore.data import (
    NestedDictionaryDataset,
    EpochShuffleDataset,
)
from vabs.data import (
    LMDB2Dataset,
    AtomPosDataset,
    ResidueDataset,
    AtomTypeDataset,
    EdgeIndexDataset,
    ResEdgeAttrDataset,
    BatchIndexDataset,
    ZipLMDB2Dataset,
    IFPreDataset,
    ClusteredDataset,
    TriEdgeIndexDataset,
    ECDataset,
)


import argparse
import warnings

from unicore.tasks import UnicoreTask, register_task
from unicore import metrics, utils


@register_task("protein_EC")
class ECLoss(UnicoreTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", metavar="FILE", help="file prefix for data")

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        # assert split in [
        #     "train",
        #     "valid",
        # ], "Not Implemented: {}!".format(split)
        print(" > Loading {} ...".format(split))
        assert self.args.cls_type in ["EC", "MF", "BP", "CC"]
        if self.args.esm_dim==1280:
            protein_path = os.path.join(self.args.data, f"{split}_esm_cls2.lmdb")
        else:
            protein_path = os.path.join(self.args.data, f"{split}_esm_cls.lmdb")
        if self.args.cls_type == "EC" and split == "train":
            remove_list = [
                11203,
                4298,
                8425,
                11939,
                1632,
            ]
        else:
            remove_list = []
        lmdb_dataset = ZipLMDB2Dataset(protein_path, remove_list=remove_list)
        self.split = split
        is_train = split == "train"
        is2re_dataset = ECDataset(
            lmdb_dataset, self.args, is_train=is_train, split=split
        )
        
        atom_pos = AtomPosDataset(is2re_dataset, "atom_pos")
        # torsion = AtomPosDataset(is2re_dataset, "torsion")
        # torsion_mask = AtomPosDataset(is2re_dataset, "torsion_mask")
        atom_type = AtomTypeDataset(is2re_dataset, "atom_type")
        if self.args.use_trimul:
            edge_index_left = TriEdgeIndexDataset(is2re_dataset, "res_edge_index_left")
            edge_index_right = TriEdgeIndexDataset(is2re_dataset, "res_edge_index_right")
        else:
            edge_index_left = AtomTypeDataset(is2re_dataset, "idx")
            edge_index_right = AtomTypeDataset(is2re_dataset, "idx")
        idxs = AtomTypeDataset(is2re_dataset, "idx")
        esm_feat = AtomPosDataset(is2re_dataset, "esm_feat")
        residue_type = ResidueDataset(is2re_dataset, "residue_type")

        atom_pos_origin = AtomPosDataset(is2re_dataset, "atom_pos_all_origin")
        atom_type_origin = AtomTypeDataset(is2re_dataset, "atom_type_origin")
        residue_type_origin = ResidueDataset(is2re_dataset, "residue_type_origin")

        edge_index = EdgeIndexDataset(is2re_dataset)

        batch_index = BatchIndexDataset(is2re_dataset)
        batch_index_res = BatchIndexDataset(is2re_dataset, "batch_index_res")
        batch_id = BatchIndexDataset(is2re_dataset, "batch_id")

        aa_edge_index =EdgeIndexDataset(is2re_dataset, "aa_edge_index")

        edge_vec = ResEdgeAttrDataset(is2re_dataset, "edge_vec") 
        edge_aa_vec = ResEdgeAttrDataset(is2re_dataset, "edge_aa_vec") 
        residue_pos_all = EdgeIndexDataset(is2re_dataset, "residue_pos_all")

        res_mask = AtomTypeDataset(is2re_dataset, "res_mask")
        atom_pos_pred_index = AtomTypeDataset(is2re_dataset, "atom_pos_pred_index")
        is_train = AtomTypeDataset(is2re_dataset, "is_train")
        label = AtomTypeDataset(is2re_dataset, "label")

        dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "residue_type": residue_type,
                    "edge_index": edge_index,
                    "atom_type": atom_type,
                    "atom_pos": atom_pos,
                    "aa_edge_index": aa_edge_index,
                    "residue_idx_all": residue_pos_all,
                    "atom_pred_pos_index": atom_pos_pred_index,
                    "edge_vec": edge_vec,
                    "edge_aa_vec": edge_aa_vec,
                    "batch_index_res": batch_index_res,
                    "res_mask": res_mask,
                    "esm_feat": esm_feat,
                    "batch_index_res": batch_id,
                    "edge_index_left": edge_index_left,
                    "edge_index_right": edge_index_right,
                    "batch_index": batch_index,
                    # "torsion_mask": torsion_mask,
                    # "torsion": torsion,
                    "idx": idxs,
                    "is_train": is_train
                },
                "targets": {
                    "atom_pos": atom_pos_origin,
                    "residue_type": residue_type_origin,
                    "batch_index_res": batch_index_res,
                    "batch_index": batch_index,
                    "atom_type": atom_type_origin,
                    "label": label,
                    # "torsion_mask": torsion_mask,
                    # "torsion": torsion,
                },
            },
        )

        if split == "train":
            dataset = EpochShuffleDataset(
                dataset,
                size=len(dataset),
                seed=self.args.seed,
            )

        print("| Loaded {} with {} samples".format(split, len(dataset)))

        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model

    def reduce_metrics(self, logging_outputs, loss, split='train'):
        """Aggregate logging outputs from data parallel training."""
        if not any("bsz" in log for log in logging_outputs):
            warnings.warn(
                "bsz not found in Loss logging outputs, cannot log bsz"
            )
        else:
            bsz = sum(log.get("bsz", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", bsz, priority=190, round=1)
        split = self.split
        loss.__class__.reduce_metrics(self, logging_outputs, split)