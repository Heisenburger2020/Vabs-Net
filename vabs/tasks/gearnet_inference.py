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
    ListDataset,
    ESMPocketDataset,
    DrugProteinDataset,
    GearPocketDataset,
    StringDataset,
    GearInferenceDataset
)
import argparse

from unicore.tasks import UnicoreTask, register_task


@register_task("gear_inference")
class GearInferenceTask(UnicoreTask):
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

        db_path = os.path.join(self.args.data, "pdbbind_test.lmdb")
        lmdb_dataset = ZipLMDB2Dataset(db_path)


        is_train = split == "train"
        is2re_dataset = GearInferenceDataset(
            lmdb_dataset, self.args, is_train=is_train
        )
        residue_type = ResidueDataset(is2re_dataset, "residue_type")
        pdb_id = StringDataset(is2re_dataset)

        protein = DrugProteinDataset(is2re_dataset, "protein")
        batch_index = BatchIndexDataset(is2re_dataset)
        residue_pos_all = EdgeIndexDataset(is2re_dataset, "residue_pos_all")
        res_mask = AtomTypeDataset(is2re_dataset, "res_mask")
        atom_type = AtomTypeDataset(is2re_dataset, "atom_type")
        input = AtomPosDataset(is2re_dataset, "input")
        dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "graph": protein,
                    "input": input,
                    "residue_idx_all": residue_pos_all,
                    "atom_type": atom_type,
                    "res_mask": res_mask,
                    "batch_index": batch_index,
                    "residue_type": residue_type,
                    "pdb_id": pdb_id,
                },
                "targets": {
                    "batch_index": batch_index,
                    # "pocket_label_all": pocket_label_all,
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