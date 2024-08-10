""" tuning of the training_model_1 stage
"""
import torch
import torch.nn as nn
import numpy as np
import albumentations as A

from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from tqdm import tqdm
import pathlib
import pickle

from src import metrics
from src.data_loader import prohibitBatchDataGetter
from src.vizualize import vizualizeSegmentation


class myProhibitBatchDataGetter(prohibitBatchDataGetter):
    def __init__(self, batch_size: int, final_img_size: tuple[int], img_ids_path: pathlib.Path, augment: A.Compose, prohibit_img_ids: list | None = None) -> None:
        """ every mask is transformed to bg/upper_body/lower_body classes only
            bg pixels on images are zeroed
        """
        super().__init__(batch_size, final_img_size, img_ids_path, augment, prohibit_img_ids)

        # load params
        with open("params_2.yaml", "r") as f:
            param_dict = yaml.full_load(f)
        self._lower_body_classes = param_dict["lower_body_classes"]
        self._upper_body_classes = param_dict["upper_body_classes"]

    def _getBatch(self, batch_ids: list):
        output_imgs, output_masks = super()._getBatch(batch_ids)

        # transform mask
        for up_body_cl in self._upper_body_classes:
            output_masks[output_masks == up_body_cl] = 1
        for low_body_cl in self._lower_body_classes:
            output_masks[output_masks == low_body_cl] = 2
        # transform imgs
        output_imgs[output_masks.unsqueeze(1).repeat(1, 3, 1, 1) == 0] = 0.0

        return output_imgs, output_masks


def getClassesWeights() -> torch.Tensor:
    """strategy for class weightening
    """
    # load params
    with open("params_2.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # load preprocess results
    with open("results/preprocess.yaml", "r") as f:
        preproc_dict = yaml.full_load(f)
    # load classes
    with open(os.environ["DATA_DIR"] + "/classes.pkl", "rb") as f:
        classes: dict = pickle.load(f)

    class_weights = torch.zeros(3, dtype=torch.float32)
    
    # define class weights based on classes freqs
    # bg class does not contribute for the model
    
    init_classes_freqs = 1 - preproc_dict["class_freq"][0]
    # upper body
    upper_body_freq = sum(
        [preproc_dict["class_freq"][up_body_cl] for up_body_cl in param_dict["upper_body_classes"]]
    )
    class_weights[1] = 1 / (upper_body_freq / init_classes_freqs)
    # lower body
    lower_body_freq = sum(
        [preproc_dict["class_freq"][up_body_cl] for up_body_cl in param_dict["lower_body_classes"]]
    )
    class_weights[2] = 1 / (lower_body_freq / init_classes_freqs)

    return class_weights

def getTrainDataLoader():
    # load params
    with open("params_2.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # load preprocess results
    with open("results/preprocess.yaml", "r") as f:
        preproc_dict = yaml.full_load(f)

    # define augmentations

    # first resize image to one resolution
    first_resize = A.Resize(*param_dict["model"]["inter_img_size"], p=1)
    # randomly apply pixel-level transformations (not used)
    pixel_level = A.Compose([
        A.Blur(p=0.2),
        A.RandomBrightnessContrast(p=0.2)
    ])
    # 
    spatial_level = A.Compose([
        # does not consider bg class
        A.CropNonEmptyMaskIfExists(150, 150, p=0.6)
    ])
    final_resize = A.Resize(*param_dict["model"]["final_img_size"], p=1)
    augment = A.Compose([final_resize])

    return myProhibitBatchDataGetter(
        param_dict["batch_size"],
        param_dict["model"]["final_img_size"],
        os.environ["DATA_DIR"] + "/train_id.txt",
        augment,
        preproc_dict["grey_img_ids"]
    )

def getImgsMasksToViz(num_exmpls: int) -> torch.Tensor:
    examples = getValDataLoader().generateBatch()
    
    return examples[0][:num_exmpls], examples[1][:num_exmpls]

def getValDataLoader():
    # load params
    with open("params_2.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # load preprocess results
    with open("results/preprocess.yaml", "r") as f:
        preproc_dict = yaml.full_load(f)

    # define augmentations
    augment = A.Compose([A.Resize(*param_dict["model"]["final_img_size"], p=1)])

    return myProhibitBatchDataGetter(
        param_dict["batch_size"],
        param_dict["model"]["final_img_size"],
        os.environ["DATA_DIR"] + "/val_id.txt",
        augment
    )

def logValMetrics(
        epoch: int,
        model: nn.Module,
        device: torch.device,
        functional,
        valid_loader,
        img_to_viz: torch.Tensor,
        img_target_masks: torch.Tensor,
        writer: SummaryWriter
) -> None:
    # load params
    with open("params_2.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # load preprocess results
    with open("results/preprocess.yaml", "r") as f:
        preproc_dict = yaml.full_load(f)

    model.eval()

    # log validate metrics
    val_loss = 0
    val_mIoU = 0
    val_accuracy = 0
    num_batches = 0

    for val_img, val_masks in tqdm(valid_loader, desc="Validation...", leave=False):
        val_img = val_img.to(device)
        val_masks = val_masks.to(device)

        val_probs: torch.Tensor = model(val_img)

        # add dummy bg prob in model answers
        val_probs = torch.concat(
            [
                torch.full([val_probs.shape[0], 1, val_probs.shape[2], val_probs.shape[3]], fill_value=float("-inf"))
                    .to(dtype=torch.float32, device=device),
                val_probs
            ],
            dim=1
        )
        val_loss += functional(val_probs, val_masks).item()
        val_probs = val_probs[:, 1:, ...]
        
        # do not consider bg class in metrics
        val_mIoU += metrics.mIoU(
                        val_probs.argmax(dim=1) + 1,
                        val_masks,
                        list(param_dict["classes"].keys()),
                        device,
                        leave_bg=True
                    ).mean().item()
        val_accuracy += metrics.Accuracy(
                            val_probs.argmax(dim=1) + 1,
                            val_masks,
                            list(param_dict["classes"].keys()),
                            device,
                            leave_bg=True
                        ).mean().item()

        num_batches += 1

    val_loss /= num_batches
    val_mIoU /= num_batches
    val_accuracy /= num_batches

    writer.add_scalar(f'{param_dict["model"]["name"]}/Validate/loss', val_loss, epoch)
    writer.add_scalar(f'{param_dict["model"]["name"]}/Validate/mIoU', val_mIoU, epoch)
    writer.add_scalar(f'{param_dict["model"]["name"]}/Validate/accuracy', val_accuracy, epoch)

    # vizualize segmentation on several examples on test
    with torch.no_grad():
        model.eval()
        # target classes start from 1
        model_mask = model(img_to_viz.detach().to(device)).argmax(dim=1) + 1
        # set bg pixels, they don't count
        model_mask[img_target_masks == 0] = 0
    for i in tqdm(range(img_to_viz.shape[0]), desc="Vizualization...", leave=False):
        fig, ax = vizualizeSegmentation(
            np.moveaxis(img_to_viz[i].numpy(), 0, 2).astype(np.int32),
            model_mask[i].cpu().numpy(),
            param_dict["classes"]
        )
        ax.set_title(f"Test example {i}")

        writer.add_figure(f'{param_dict["model"]["name"]}/Validate/segmentation/expl_{i}', fig, epoch)
