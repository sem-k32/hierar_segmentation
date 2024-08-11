""" utilities for training model
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

from src import metrics
from src.data_loader import prohibitBatchDataGetter
from src.vizualize import vizualizeSegmentation


class myProhibitBatchDataGetter(prohibitBatchDataGetter):
    def __init__(self, batch_size: int, final_img_size: tuple[int], img_ids_path: pathlib.Path, augment: A.Compose, prohibit_img_ids: list | None = None) -> None:
        """ every mask is additionally transformed to bg/human classes only
        """
        super().__init__(batch_size, final_img_size, img_ids_path, augment, prohibit_img_ids)

    def _getBatch(self, batch_ids: list):
        output_imgs, output_masks = super()._getBatch(batch_ids)
        output_masks = (output_masks != 0).to(dtype=torch.long)

        return output_imgs, output_masks


def getClassesWeights() -> torch.Tensor:
    """strategy for class weightening
    """
    # results dirs
    result_dir = pathlib.Path("../results")
    # load preprocess results
    with open(result_dir / "preprocess.yaml", "r") as f:
        preproc_dict = yaml.full_load(f)

    class_weights = torch.empty(len(param_dict["classes"]), dtype=torch.float32)
    
    # define class weights based on classes freqs

    # bg
    class_weights[0] = 1 / preproc_dict["class_freq"][0]
    # human
    class_weights[1] = 1 / (1 - preproc_dict["class_freq"][0])

    return class_weights

def getTrainDataLoader():
    # load params
    with open("params.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # results dirs
    result_dir = pathlib.Path("../results")
    # load preprocess results
    with open(result_dir / "preprocess.yaml", "r") as f:
        preproc_dict = yaml.full_load(f)

    # define augmentations

    # first resize image to one resolution
    first_resize = A.Resize(*param_dict["model"]["inter_img_size"], p=1)
    # randomly apply pixel-level transformations
    pixel_level = A.Compose([
        A.Blur(p=0.2),
        A.RandomBrightnessContrast(p=0.2)
    ])
    # 
    spatial_level = A.Compose([
        A.RandomSizedCrop((100, 300), size=param_dict["model"]["final_img_size"], p=0.6)
    ])
    final_resize = A.Resize(*param_dict["model"]["final_img_size"], p=1)
    augment = A.Compose([first_resize, pixel_level, spatial_level, final_resize])

    return myProhibitBatchDataGetter(
        param_dict["batch_size"],
        param_dict["model"]["final_img_size"],
        os.environ["DATA_DIR"] + "/train_id.txt",
        augment,
        preproc_dict["grey_img_ids"]
    )

def getImgsToViz(num_exmpls: int) -> torch.Tensor:
    val_loader = getValDataLoader()
    
    return val_loader.generateBatch()[0][:num_exmpls]


def getValDataLoader():
    # load params
    with open("params.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # results dirs
    result_dir = pathlib.Path("../results")
    # load preprocess results
    with open(result_dir / "preprocess.yaml", "r") as f:
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
        writer: SummaryWriter
) -> None:
    # load params
    with open("params.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # results dirs
    result_dir = pathlib.Path("../results")
    # load preprocess results
    with open(result_dir / "preprocess.yaml", "r") as f:
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

        val_loss += functional(val_probs, val_masks).item()
        val_mIoU += metrics.mIoU(
                        val_probs.argmax(dim=1),
                        val_masks,
                        list(param_dict["classes"].keys()),
                        device
                    ).mean().item()
        val_accuracy += metrics.Accuracy(
                            val_probs.argmax(dim=1),
                            val_masks,
                            list(param_dict["classes"].keys()),
                            device
                        ).mean().item()

        num_batches += 1

    val_loss /= num_batches
    val_mIoU /= num_batches
    val_accuracy /= num_batches

    writer.add_scalar("Validate/loss", val_loss, epoch)
    writer.add_scalar("Validate/mIoU", val_mIoU, epoch)
    writer.add_scalar("Validate/accuracy", val_accuracy, epoch)

    # vizualize segmentation on several examples on test
    with torch.no_grad():
        model.eval()
        model_mask = model(img_to_viz.detach().to(device)).argmax(dim=1)
    for i in tqdm(range(img_to_viz.shape[0]), desc="Vizualization...", leave=False):
        fig, ax = vizualizeSegmentation(
            np.moveaxis(img_to_viz[i].numpy(), 0, 2).astype(np.int32),
            model_mask[i].cpu().numpy(),
            param_dict["classes"]
        )
        ax.set_title(f"Test example {i}")

        writer.add_figure(f"Validate/segmentation/expl_{i}", fig, epoch)
