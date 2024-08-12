""" data loader classes for iteration/random batch generation of images and masks
"""
import PIL.Image as Image
import numpy as np
import PIL
import albumentations as A
import torch

import pathlib
import os
from copy import deepcopy
from typing import Optional


class batchDataGetter:
    def __init__(
        self,
        batch_size: int,
        final_img_size: tuple[int],
        img_ids_path: pathlib.Path,
        augment: A.Compose
    ) -> None:
        """access to batched images/masks from disk with given augmentation transforms

        Args:
            batch_size (int): 
            final_img_size (tuple[int]): image size for model input
            img_ids_path (pathlib.Path):
            augment (A.Compose): image transformations
        """
        self.batch_size = batch_size
        self._final_img_size = final_img_size
        self._augment = augment
        
        self._img_dir = pathlib.Path(os.environ["DATA_DIR"] + "/JPEGImages")
        self._mask_dir = pathlib.Path(os.environ["DATA_DIR"] + "/gt_masks")
        
        # get images id
        with open(img_ids_path, "r") as f:
            self._img_ids = f.readlines()
            self._img_ids = list(map(lambda id: id.replace("\n", ""), self._img_ids))
            self._img_ids = np.array(list(self._img_ids))

    def generateBatch(self) -> tuple[torch.Tensor]:
        batch_ids = list(np.random.choice(self._img_ids, self.batch_size, replace=False))
        return self._getBatch(batch_ids)
        
    def __iter__(self):
        self._img_ids_iter_list = deepcopy(self._img_ids.tolist())

        return self
    
    def __next__(self) -> tuple[torch.Tensor]:
        if len(self._img_ids_iter_list) == 0:
            del self._img_ids_iter_list
            raise StopIteration()
        
        # compute batch size
        cur_batch_size = min(len(self._img_ids_iter_list), self.batch_size)

        batch_ids = self._img_ids_iter_list[:cur_batch_size]
        # remove this ids from the queue
        self._img_ids_iter_list = self._img_ids_iter_list[cur_batch_size:]
        
        return self._getBatch(batch_ids)

    def _getBatch(self, batch_ids: list) -> tuple[torch.Tensor]:
        output_imgs = torch.FloatTensor(len(batch_ids), 3, *self._final_img_size)
        output_masks = torch.LongTensor(len(batch_ids), *self._final_img_size)

        for i, img_id in enumerate(batch_ids):
            # read img and mask
            img = np.array(Image.open(self._img_dir / f"{img_id}.jpg"))
            mask = np.load(self._mask_dir / f"{img_id}.npy")
            # use augmentations
            transformed = self._augment(image=img, mask=mask)
            img = transformed["image"]
            img = np.moveaxis(img, 2, 0)
            mask = transformed["mask"]

            output_imgs[i] = torch.from_numpy(img).to(dtype=torch.float32)
            output_masks[i] = torch.from_numpy(mask).to(dtype=torch.long)

        return output_imgs, output_masks


class prohibitBatchDataGetter(batchDataGetter):
    def __init__(self, batch_size: int, final_img_size: tuple[int], img_ids_path: pathlib.Path, augment: A.Compose, prohibit_img_ids: Optional[list] = None) -> None:
        """ allows to exclude some images from data loader

        Args:
            prohibit_img_ids (Optional[list], optional): list of exluded image ids. Defaults to None.
        """
        super().__init__(batch_size, final_img_size, img_ids_path, augment)

        # remove prohibited img ids
        if prohibit_img_ids is not None:
            self._img_ids = set(self._img_ids.tolist()).difference(set(prohibit_img_ids))
            self._img_ids = np.array(list(self._img_ids))
