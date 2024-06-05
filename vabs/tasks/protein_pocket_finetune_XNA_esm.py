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
)
import argparse

from unicore.tasks import UnicoreTask, register_task


@register_task("protein_pocket_ft_xna_esm")
class ESMXNAProteinftTask(UnicoreTask):
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
            protein_path = os.path.join(self.args.data, f"{self.args.pocket_type}_train_esm_cls.lmdb")
            lmdb_dataset = ZipLMDB2Dataset(protein_path)
        elif split == "valid":
            assert 0, "no valid"
            if not self.args.use_clean:
                db_path = os.path.join(self.args.data, "PointSiteDataset_valid_long_with_esm_cpu.lmdb")
            else:
                db_path = os.path.join(self.args.data, "clean_single_with_esm_cpu.lmdb")

            # db_path = os.path.join(self.args.data, "valid_filter_alphaC.lmdb")
            lmdb_dataset = LMDB2Dataset(db_path)
        else:
            db_path = os.path.join(self.args.data, f"{self.args.pocket_type}_test_esm_cls.lmdb")
            # db_path = os.path.join(self.args.data, "valid_filter_alphaC.lmdb")
            lmdb_dataset = ZipLMDB2Dataset(db_path)


        is_train = split == "train"
        is2re_dataset = ESMPocketDataset(
            lmdb_dataset, self.args, is_train=is_train
        )
        
        list_str = ListDataset(is2re_dataset, "res_str")
        batch_index = BatchIndexDataset(is2re_dataset)

        pocket_label_all = PocketTaskDataset(is2re_dataset, "pocket_label_all")
        dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "list_str": list_str,
                    "batch_index": batch_index,
                },
                "targets": {
                    "batch_index": batch_index,
                    "pocket_label_all": pocket_label_all,
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