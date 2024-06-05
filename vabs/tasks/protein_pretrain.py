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
    ClusteredMergeDataset
)


import argparse

from unicore.tasks import UnicoreTask, register_task


@register_task("protein_pretrain")
class ProteinTask(UnicoreTask):
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

        if split == "train":

            if self.args.use_esm_feat:
                protein_path = r"./store/dataset/esm2"
                lmdb_dataset = ClusteredMergeDataset(protein_path)
            else:
                protein_path = os.path.join(self.args.data, "train_merged_clustered.lmdb")
                lmdb_dataset = ClusteredDataset(protein_path)
            print("dataset:", protein_path)
        else:
            # db_path = os.path.join(self.args.data, "valid.lmdb")
            db_path = os.path.join(self.args.data, "valid.lmdb")
            lmdb_dataset = ZipLMDB2Dataset(db_path)

        is_train = split == "train"
        is2re_dataset = IFPreDataset(
            lmdb_dataset, self.args, is_train=is_train, split=split
        )
        
        esm_feat = AtomPosDataset(is2re_dataset, "esm_feat")
        atom_pos = AtomPosDataset(is2re_dataset, "atom_pos")
        torsion = AtomPosDataset(is2re_dataset, "torsion")
        torsion_mask = AtomPosDataset(is2re_dataset, "torsion_mask")
        atom_type = AtomTypeDataset(is2re_dataset, "atom_type")
        edge_index_left = AtomTypeDataset(is2re_dataset, "idx")
        edge_index_right = AtomTypeDataset(is2re_dataset, "idx")
        idxs = AtomTypeDataset(is2re_dataset, "idx")
        residue_type = ResidueDataset(is2re_dataset, "residue_type")

        atom_pos_origin = AtomPosDataset(is2re_dataset, "atom_pos_all_origin")
        atom_type_origin = AtomTypeDataset(is2re_dataset, "atom_type_origin")
        residue_type_origin = ResidueDataset(is2re_dataset, "residue_type_origin")

        edge_index = EdgeIndexDataset(is2re_dataset)
        if not self.args.residue_only:
            aa_edge_index =EdgeIndexDataset(is2re_dataset, "aa_edge_index")
        else:
            aa_edge_index = EdgeIndexDataset(is2re_dataset)

        batch_index = BatchIndexDataset(is2re_dataset)
        batch_index_res = BatchIndexDataset(is2re_dataset, "batch_index_res")
        batch_id = BatchIndexDataset(is2re_dataset, "batch_id")


        edge_vec = ResEdgeAttrDataset(is2re_dataset, "edge_vec") 
        if not self.args.residue_only:
            edge_aa_vec = ResEdgeAttrDataset(is2re_dataset, "edge_aa_vec") 
        else:
            edge_aa_vec = ResEdgeAttrDataset(is2re_dataset, "edge_vec") 

        residue_pos_all = EdgeIndexDataset(is2re_dataset, "residue_pos_all")

        res_mask = AtomTypeDataset(is2re_dataset, "res_mask")
        atom_pos_pred_index = AtomTypeDataset(is2re_dataset, "atom_pos_pred_index")
        is_train = AtomTypeDataset(is2re_dataset, "is_train")
        gb_feat = AtomTypeDataset(is2re_dataset, "is_train")
        atom_sas = AtomTypeDataset(is2re_dataset, "atom_sas")
        dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "residue_type": residue_type,
                    "edge_index": edge_index,
                    "atom_type": atom_type,
                    "atom_pos": atom_pos,
                    "aa_edge_index": aa_edge_index,
                    "esm_feat": esm_feat,
                    "residue_idx_all": residue_pos_all,
                    "atom_pred_pos_index": atom_pos_pred_index,
                    "edge_vec": edge_vec,
                    "edge_aa_vec": edge_aa_vec,
                    "batch_index_res": batch_index_res,
                    "res_mask": res_mask,
                    "batch_index": batch_index,
                    "batch_index_res": batch_id,
                    "edge_index_left": edge_index_left,
                    "edge_index_right": edge_index_right,
                    "torsion_mask": torsion_mask,
                    "torsion": torsion,
                    "idx": idxs,
                    "gb_feat": gb_feat,
                    "is_train": is_train
                },
                "targets": {
                    "atom_pos": atom_pos_origin,
                    "residue_type": residue_type_origin,
                    "batch_index_res": batch_index_res,
                    "batch_index": batch_index,
                    "atom_sas": atom_sas,
                    "atom_type": atom_type_origin,
                    "torsion_mask": torsion_mask,
                    "torsion": torsion,
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

    def end_epoch(self, epoch, model):
        """Hook function called after the end of each epoch."""
        pass

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model
