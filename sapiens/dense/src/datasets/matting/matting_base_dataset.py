# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import List

import numpy as np
from sapiens.engine.datasets import BaseDataset
from sapiens.registry import DATASETS
from typedio.auto import typed


##-----------------------------------------------------------------------
@DATASETS.register_module()
class MattingBaseDataset(BaseDataset):
    def __init__(self, ann_file, **kwargs) -> None:
        self.ann_file = ann_file
        super().__init__(**kwargs)
        return

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.
        modify this function to load a list of all sample information in training

        Returns:
            list[dict]: All data info of dataset.
        """
        raw_data = typed.load(self.ann_file)

        data_list = [
            {
                "image_path": os.path.join(self.data_root, sample["image"]),
                "mask_path": os.path.join(self.data_root, sample["mask"]),
            }
            for sample in raw_data
        ]

        print(
            "\033[92mDone! {}. Loaded total samples: {}. Test mode: {}\033[0m".format(
                self.__class__.__name__, len(data_list), self.test_mode
            )
        )

        return data_list

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])

        try:
            img = typed.load(data_info["image_path"])  # rgb
            img = img[..., [2, 1, 0]]  # rgb -> bgr

            mask = typed.load(data_info["mask_path"]).astype(np.float32)
        except Exception as e:
            print(f"Error loading image/mask {data_info}: {e}")
            return None

        fgr = None
        if mask.ndim == 3 and mask.shape[-1] == 4:
            fgr = mask[..., [2, 1, 0]]  # rgb -> bgr
            alpha = mask[..., -1] / 255.0

            fgr = (fgr * alpha[..., None]).astype(np.uint8)  # pre-multiply alpha
        elif mask.ndim == 3 and (mask.shape[-1] == 3 or mask.shape[-1] == 1):
            # mask could be HxWx3 with same values in each channel
            alpha = mask[..., 0] / 255.0
        elif mask.ndim == 2:
            alpha = mask / 255.0
        else:
            print(f"Unexpected mask shape {mask.shape} for {data_info}")
            return None

        data_info = {
            "img": img,
            "img_id": "",
            "img_path": data_info["image_path"],
            "alpha": alpha,
            "id": idx,
        }

        if fgr is not None:
            data_info["fgr"] = fgr

        return data_info
