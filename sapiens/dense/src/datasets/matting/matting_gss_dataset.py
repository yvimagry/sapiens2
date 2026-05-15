# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import List

import numpy as np
from sapiens.registry import DATASETS
from typedio.auto import typed

from .matting_base_dataset import MattingBaseDataset


##-----------------------------------------------------------------------
@DATASETS.register_module()
class MattingGSSDataset(MattingBaseDataset):
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.
        modify this function to load a list of all sample information in training

        Returns:
            list[dict]: All data info of dataset.
        """
        raw_data = typed.load(self.ann_file)
        image_paths = raw_data.split("\n")

        data_list = []
        for file_path in image_paths:
            if not file_path:
                continue

            data_list.append(
                {
                    "image_path": os.path.join(self.data_root, file_path.strip()),
                }
            )

        print(
            "\033[92mDone! {}. Loaded total samples: {}. Test mode: {}\033[0m".format(
                self.__class__.__name__, len(data_list), self.test_mode
            )
        )

        return data_list

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])

        img = typed.load(data_info["image_path"])  # RGBA
        #  GSS images are 16-bit pngs, so we need to convert them to 8-bit
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)

        bgr = img[..., [2, 1, 0]]  # RGB to BGR

        mask = img[:, :, 3].astype(np.float32) / 255.0
        fgr = (bgr.astype(np.float32) * mask[..., None]).astype(np.uint8)

        data_info = {
            "img": bgr,
            "img_id": "",
            "img_path": data_info["image_path"],
            "alpha": mask,
            "fgr": fgr,
            "id": idx,
        }

        return data_info
