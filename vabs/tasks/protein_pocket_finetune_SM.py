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
    PocketDataset,
    AtomPosDataset,
    PocketTaskDataset,
    ResidueDataset,
    AtomTypeDataset,
    EdgeIndexDataset,
    ResEdgeAttrDataset,
    BatchIndexDataset,
    ZipLMDB2Dataset,
)
import argparse

from unicore.tasks import UnicoreTask, register_task


@register_task("protein_pocket_ft")
class ProteinftTask(UnicoreTask):
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

        print(" > Loading {} ...".format(split))

        if split == "train":
            protein_path = os.path.join(self.args.data, "train_filtered_with_esm_cpu_esm_train2.lmdb")
            lmdb_dataset = ZipLMDB2Dataset(protein_path)
        elif split == "valid":
            protein_path = os.path.join(self.args.data, "train_filtered_with_esm_cpu_esm_valid2.lmdb")
            lmdb_dataset = ZipLMDB2Dataset(protein_path)
        else:
            db_path = os.path.join(self.args.data, f"{split}_with_esm_cpu2.lmdb")
            # db_path = os.path.join(self.args.data, f"{split}_with_esm_cpu_esm.lmdb")
            # db_path = os.path.join(self.args.data, "valid_filter_alphaC.lmdb")
            lmdb_dataset = ZipLMDB2Dataset(db_path)


        is_train = split == "train"
        is2re_dataset = PocketDataset(
            lmdb_dataset, self.args, is_train=is_train
        )
        
        atom_pos = AtomPosDataset(is2re_dataset, "atom_pos")
        atom_type = AtomTypeDataset(is2re_dataset, "atom_type")

        idxs = AtomTypeDataset(is2re_dataset, "idx")
        residue_type = ResidueDataset(is2re_dataset, "residue_type")

        atom_pos_origin = AtomPosDataset(is2re_dataset, "atom_pos_all_origin")
        atom_type_origin = AtomTypeDataset(is2re_dataset, "atom_type_origin")
        residue_type_origin = ResidueDataset(is2re_dataset, "residue_type_origin")
        edge_index_left = AtomTypeDataset(is2re_dataset, "idx")
        edge_index_right = AtomTypeDataset(is2re_dataset, "idx")
        edge_index = EdgeIndexDataset(is2re_dataset)
        esm_feat = AtomPosDataset(is2re_dataset, "esm_feat")

        batch_index = BatchIndexDataset(is2re_dataset)
        batch_index_res = BatchIndexDataset(is2re_dataset, "batch_index_res")

        aa_edge_index =EdgeIndexDataset(is2re_dataset, "aa_edge_index")

        edge_vec = ResEdgeAttrDataset(is2re_dataset, "edge_vec") 
        edge_aa_vec = ResEdgeAttrDataset(is2re_dataset, "edge_aa_vec") 

        res_mask = AtomTypeDataset(is2re_dataset, "res_mask")
        residue_pos = EdgeIndexDataset(is2re_dataset, "residue_pos")
        residue_pos_all = EdgeIndexDataset(is2re_dataset, "residue_pos_all")

        if self.args.use_sas:
            atom_sas = AtomTypeDataset(is2re_dataset, "atom_sas")
        else:
            atom_sas = AtomTypeDataset(is2re_dataset, "idx")
        pocket_label_all = PocketTaskDataset(is2re_dataset, "pocket_label_all")
        pocket_label = PocketTaskDataset(is2re_dataset, "label")
        dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "residue_type": residue_type,
                    "edge_index": edge_index,
                    "residue_idx_all": residue_pos_all,
                    "residue_idx": residue_pos,
                    "atom_type": atom_type,
                    "atom_pos": atom_pos,
                    "aa_edge_index": aa_edge_index,
                    "edge_vec": edge_vec,
                    "edge_aa_vec": edge_aa_vec,
                    "esm_feat": esm_feat,
                    "atom_sas": atom_sas,
                    "res_mask": res_mask,
                    "edge_index_left": edge_index_left,
                    "edge_index_right": edge_index_right,
                    "batch_index_res": batch_index_res,
                    "batch_index": batch_index,
                    "idx": idxs,
                },
                "targets": {
                    "atom_pos": atom_pos_origin,
                    "residue_type": residue_type_origin,
                    "batch_index": batch_index,
                    "batch_index_res": batch_index_res,
                    "label": pocket_label,
                    "pocket_label_all": pocket_label_all,
                    "residue_pos_all": residue_pos_all,
                    "atom_type": atom_type_origin,
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