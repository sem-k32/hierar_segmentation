import PIL.Image as Image
import numpy as np
import PIL
import albumentations as A
import torch

import pathlib
import os


def getTrainBatchLoader(batch_size: int, prohibit_images_id: list):
    """ implements random batched train data loader with augmentations
    """
    def batchLoader():
        # read root of the working directory
        ws_dir = pathlib.Path(os.environ["WORKSPACE_DIR"])
        # model will receive such images
        MODEL_IMG_SIZE = (256, 256)

        # get train images id
        with open(ws_dir / "Pascal-part/train_id.txt", "r") as f:
            train_ids = f.readlines()
            train_ids = set(map(lambda id: id.replace("\n", ""), train_ids))
            train_ids = train_ids.difference(set(prohibit_images_id))
            train_ids = np.array(list(train_ids))

        # augmentations pipeline

        # first resize image to one resolution
        first_resize = A.Resize(300, 300, p=1)
        # randomly apply pixel-level transformations
        pixel_level = A.Compose([
            A.Blur(p=0.2),
            A.RandomBrightnessContrast(p=0.2)
        ])
        # 
        spatial_level = A.Compose([
            A.RandomSizedCrop((100, 300), size=MODEL_IMG_SIZE, p=0.6)
        ])
        final_resize = A.Resize(MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[0], p=1)
        augment = A.Compose([first_resize, pixel_level, spatial_level, final_resize])

        # generate batches
        while True:
            output_imgs = torch.FloatTensor(batch_size, 3, MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[1])
            output_masks = torch.LongTensor(batch_size, MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[1])

            batch_ids = np.random.choice(train_ids, batch_size, replace=False)
            for i, img_id in enumerate(batch_ids):
                # read img and mask
                img = np.array(Image.open(ws_dir / f"Pascal-part/JPEGImages/{img_id}.jpg"))
                mask = np.load(ws_dir / f"Pascal-part/gt_masks/{img_id}.npy")
                # use augmentations
                transformed = augment(image=img, mask=mask)
                img = transformed["image"]
                img = np.moveaxis(img, 2, 0)
                mask = transformed["mask"]

                output_imgs[i] = torch.from_numpy(img).to(dtype=torch.float32)
                output_masks[i] = torch.from_numpy(mask).to(dtype=torch.long)

            yield output_imgs, output_masks
    
    return batchLoader


class testDataIter:
    def __init__(self, batch_size: int):
        """ implements batch iterator through validate images/masks.
             Images are resized to the model input size
        """
        self._batch_size = batch_size

    def __iter__(self):
         # read root of the working directory
        self._ws_dir = pathlib.Path(os.environ["WORKSPACE_DIR"])
        # model will receive such images
        self._MODEL_IMG_SIZE = (256, 256)

        # get test images id
        with open(self._ws_dir / "Pascal-part/val_id.txt", "r") as f:
            self._validate_ids = f.readlines()
            self._validate_ids = list(map(lambda id: id.replace("\n", ""), self._validate_ids))

        # resize augmentation for model fit
        self._resize_augment = A.Resize(self._MODEL_IMG_SIZE[0], self._MODEL_IMG_SIZE[1])

        return self
    
    def __next__(self) -> tuple[torch.Tensor]:
        if len(self._validate_ids) == 0:
            raise StopIteration()
        
        # compute batch size
        cur_batch_size = min(len(self._validate_ids), self._batch_size)
        # output containers
        output_imgs = torch.empty((cur_batch_size, 3, self._MODEL_IMG_SIZE[0], self._MODEL_IMG_SIZE[1]), 
                                  dtype=torch.float32)
        output_masks = torch.empty((cur_batch_size, self._MODEL_IMG_SIZE[0], self._MODEL_IMG_SIZE[1]),
                                   dtype=torch.long)

        for i in range(cur_batch_size):
            img_id = self._validate_ids[i]
            # read img and mask
            img = np.array(Image.open(self._ws_dir / f"Pascal-part/JPEGImages/{img_id}.jpg")) 
            mask = np.load(self._ws_dir / f"Pascal-part/gt_masks/{img_id}.npy")

            # use augmentations
            transformed = self._resize_augment(image=img, mask=mask)
            img = transformed["image"]
            img = np.moveaxis(img, 2, 0)
            mask = transformed["mask"]

            output_imgs[i] = torch.from_numpy(img).to(dtype=torch.float32)
            output_masks[i] = torch.from_numpy(mask).to(dtype=torch.long)

        # remove output images from queue
        self._validate_ids = self._validate_ids[cur_batch_size:]

        return output_imgs, output_masks
