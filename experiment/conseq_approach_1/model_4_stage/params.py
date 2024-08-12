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
import pickle

from src import metrics
from src.data_loader import prohibitBatchDataGetter
from src.vizualize import vizualizeSegmentation


class myProhibitBatchDataGetter(prohibitBatchDataGetter):
    def __init__(self, batch_size: int, final_img_size: tuple[int], img_ids_path: pathlib.Path, augment: A.Compose, prohibit_img_ids: list | None = None) -> None:
        """ every mask is transformed to bg/upper_body_classes only
            bg pixels on images are zeroed
        """
        super().__init__(batch_size, final_img_size, img_ids_path, augment, prohibit_img_ids)

        # load params
        with open("params.yaml", "r") as f:
            param_dict = yaml.full_load(f)
        # load classes
        with open(os.environ["DATA_DIR"] + "/classes.pkl", "rb") as f:
            classes: dict = pickle.load(f)
            # reverse classes dict
            classes = {val: key for key, val in classes.items()}

        # define mapping from initial class indxs to model's class indxs
        self._to_model_classes = {}
        for cl_id, cl_name in param_dict["classes"].items():
            if cl_name == "bg":
                continue

            self._to_model_classes[classes[cl_name]] = cl_id

    def _getBatch(self, batch_ids: list):
        output_imgs, output_masks = super()._getBatch(batch_ids)

        # set irrelevant classes to bg
        output_masks[
            ~torch.isin(output_masks, torch.LongTensor(list(self._to_model_classes.keys())))
        ] = 0
        # transform mask target classes
        for init_cl_indx, model_cl_indx in self._to_model_classes.items():
            output_masks[output_masks == init_cl_indx] = model_cl_indx
        # transform imgs
        output_imgs[output_masks.unsqueeze(1).repeat(1, 3, 1, 1) == 0] = 0.0

        return output_imgs, output_masks


def getClassesWeights() -> torch.Tensor:
    """strategy for class weightening
    """
    # load params
    with open("params.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # results dirs
    result_dir = pathlib.Path("../results")
    # load preprocess results
    with open(result_dir / "preprocess.yaml", "r") as f:
        preproc_dict = yaml.full_load(f)
    # load classes
    with open(os.environ["DATA_DIR"] + "/classes.pkl", "rb") as f:
        classes: dict = pickle.load(f)
        # reverse classes dict
        classes = {val: key for key, val in classes.items()}

    # define mapping from model's class indxs to initial class indxs to
    to_init_classes = {}
    for cl_id, cl_name in param_dict["classes"].items():
        if cl_name == "bg":
            continue

        to_init_classes[cl_id] = classes[cl_name]

    class_weights = torch.zeros(len(param_dict["classes"]), dtype=torch.float32)
    
    # define class weights based on classes freqs
    
    overall_classes_freqs = 0
    for model_cl, init_cl in to_init_classes.items():
        overall_classes_freqs += preproc_dict["class_freq"][init_cl]
        class_weights[model_cl] = preproc_dict["class_freq"][init_cl]
    class_weights /= overall_classes_freqs
    class_weights = 1 / class_weights
    # bg class does not contribute for the model
    class_weights[0] = 0

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
    # randomly apply pixel-level transformations (not used)
    pixel_level = A.Compose([
        A.Blur(p=0.2),
        A.RandomBrightnessContrast(p=0.2)
    ])
    # 
    spatial_level = A.Compose([
        # does not consider bg class
        A.CropNonEmptyMaskIfExists(100, 100, p=0.2)
    ])
    final_resize = A.Resize(*param_dict["model"]["final_img_size"], p=1)
    augment = A.Compose([final_resize])

    # prohibit grey and irrelative for current classes imgs
    prohibit_list = preproc_dict["grey_img_ids"] + filterIrreleventTrainImgs(param_dict["irrelative_pix_threshold"])

    return myProhibitBatchDataGetter(
        param_dict["batch_size"],
        param_dict["model"]["final_img_size"],
        os.environ["DATA_DIR"] + "/train_id.txt",
        augment,
        prohibit_list
    )

def getImgsMasksToViz(num_exmpls: int) -> torch.Tensor:
    """ randomly choose several val images and their masks for segmentation vizualization

    Args:
        num_exmpls (int): number of images to pick
    """
    examples = getValDataLoader().generateBatch()
    
    return examples[0][:num_exmpls], examples[1][:num_exmpls]

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
        img_target_masks: torch.Tensor,
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
        # remove dummy bg prob
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

    writer.add_scalar('Validate/loss', val_loss, epoch)
    writer.add_scalar('Validate/mIoU', val_mIoU, epoch)
    writer.add_scalar('Validate/accuracy', val_accuracy, epoch)

    # vizualize segmentation on several examples on test

    # target classes start from 1
    model_mask = model(img_to_viz.detach().to(device)).argmax(dim=1) + 1
    # set bg pixels in model's mask
    model_mask[img_target_masks == 0] = 0
    for i in tqdm(range(img_to_viz.shape[0]), desc="Vizualization...", leave=False):
        fig, ax = vizualizeSegmentation(
            np.moveaxis(img_to_viz[i].numpy(), 0, 2).astype(np.int32),
            model_mask[i].cpu().numpy(),
            param_dict["classes"]
        )
        ax.set_title(f"Test example {i}")

        writer.add_figure(f'Validate/segmentation/expl_{i}', fig, epoch)

def filterIrreleventTrainImgs(pixels_threshold: int) -> list:
    """ exclude train images that have too little relative pixels
    """
    # load params
    with open("params.yaml", "r") as f:
        param_dict = yaml.full_load(f)

    # get initial ids of relevant classes
    with open(os.environ["DATA_DIR"] + "/classes.pkl", "rb") as f:
        classes: dict = pickle.load(f)
        # reverse classes dict
        classes = {val: key for key, val in classes.items()}

    init_cl_indxs = []
    for cl_name in param_dict["classes"].values():
            if cl_name == "bg":
                continue

            init_cl_indxs.append(classes[cl_name])


    # containers for irrelevent imgs
    irrel_img_list = []

    # get train images id
    data_dir = pathlib.Path(os.environ["DATA_DIR"])
    with open(data_dir / "train_id.txt", "r") as f:
        train_ids = f.readlines()
        train_ids = list(map(lambda id: id.replace("\n", ""), train_ids))

    mask_path = data_dir / "gt_masks"

    # filter train images
    for train_id in train_ids:
        mask = np.load(mask_path / f"{train_id}.npy")

        if np.isin(mask, init_cl_indxs).sum() < pixels_threshold:
            irrel_img_list.append(train_id)

    return irrel_img_list

