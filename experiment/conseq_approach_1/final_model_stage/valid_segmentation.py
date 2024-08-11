""" building hierarchal segmentation model
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import PIL.Image as Image
import albumentations as A

import yaml
import pickle
from tqdm import tqdm
import functools
import pathlib
import os
from datetime import datetime

from src import metrics
from src.vizualize import vizualizeSegmentation
from hierarchal_model import hierarBodySegmentator
import experiment.conseq_approach_1.model_1_stage.model as model_1
import experiment.conseq_approach_1.model_2_stage.model as model_2
import experiment.conseq_approach_1.model_3_stage.model as model_3
import experiment.conseq_approach_1.model_4_stage.model as model_4


if __name__ == "__main__":
    # load params
    with open("params.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # results dirs
    result_dir = pathlib.Path("../results")
    # load inital classes
    with open(os.environ["DATA_DIR"] + "/classes.pkl", "rb") as f:
        cl_to_name: dict = pickle.load(f)
        name_to_cl = {val: key for key, val in cl_to_name.items()}

    # device for validation
    device = torch.device("cpu")
    # torch.cuda.empty_cache()

    # compound models and their classes for final segmentator

    models_dict = {}
    models_class_to_name = {}

    # bg_human
    with open("../model_1_stage/params.yaml", "r") as f:
        models_params =  yaml.full_load(f)
        models_class_to_name["bg_human"] = models_params["classes"]

    model = model_1.directSegmentator( 
        models_params["model"]["num_levels"], 
        len(models_params["classes"]),
        models_params["model"]["kernal_size"],
        models_params["model"]["num_conv_layers"],
        models_params["model"]["encoder_dropout_p"],
        models_params["model"]["leaky_relu_slope"]
    )
    model.load_state_dict(torch.load(result_dir / "model_1.pkl", weights_only=True))
    model.to(device)
    models_dict["bg_human"] = model

    # low_up_body
    with open("../model_2_stage/params.yaml", "r") as f:
        models_params =  yaml.full_load(f)
        models_class_to_name["low_up_body"] = models_params["classes"]

    model = model_2.directSegmentator( 
        models_params["model"]["num_levels"], 
        len(models_params["classes"]) - 1,
        models_params["model"]["kernal_size"],
        models_params["model"]["num_conv_layers"],
        models_params["model"]["encoder_dropout_p"],
        models_params["model"]["leaky_relu_slope"]
    )
    model.load_state_dict(torch.load(result_dir / "model_2.pkl", weights_only=True))
    model.to(device)
    models_dict["low_up_body"] = model

    # up_body
    with open("../model_3_stage/params.yaml", "r") as f:
        models_params =  yaml.full_load(f)
        models_class_to_name["up_body"] = models_params["classes"]

    model = model_3.directSegmentator( 
        models_params["model"]["num_levels"], 
        len(models_params["classes"]) - 1,
        models_params["model"]["kernal_size"],
        models_params["model"]["num_conv_layers"],
        models_params["model"]["encoder_dropout_p"],
        models_params["model"]["leaky_relu_slope"]
    )
    model.load_state_dict(torch.load(result_dir / "model_3.pkl", weights_only=True))
    model.to(device)
    models_dict["up_body"] = model

    # low_body
    with open("../model_4_stage/params.yaml", "r") as f:
        models_params =  yaml.full_load(f)
        models_class_to_name["low_body"] = models_params["classes"]

    model = model_4.directSegmentator( 
        models_params["model"]["num_levels"], 
        len(models_params["classes"]) - 1,
        models_params["model"]["kernal_size"],
        models_params["model"]["num_conv_layers"],
        models_params["model"]["encoder_dropout_p"],
        models_params["model"]["leaky_relu_slope"]
    )
    model.load_state_dict(torch.load(result_dir / "model_4.pkl", weights_only=True))
    model.to(device)
    models_dict["low_body"] = model

    segmentator = hierarBodySegmentator(models_dict, models_class_to_name)
    # save final model
    with open(result_dir / "hierar_model.pkl", "wb") as f:
        pickle.dump(segmentator, f)

    # resize transformation for input images
    augment = A.Compose([A.Resize(*param_dict["model"]["input_img_size"], p=1)])

    # get val images id
    with open(os.environ["DATA_DIR"] + "/val_id.txt", "r") as f:
        img_ids = f.readlines()
        img_ids = list(map(lambda id: id.replace("\n", ""), img_ids))
        img_ids = np.array(img_ids)

    img_dir = pathlib.Path(os.environ["DATA_DIR"]) / "JPEGImages"
    mask_dir = pathlib.Path(os.environ["DATA_DIR"]) / "gt_masks"

    # vizualize several segmentaions segmentation

    # segm results dir
    segm_dir = result_dir / "val_segm"
    segm_dir.mkdir(exist_ok=True)

    img_ids_to_viz = np.random.choice(img_ids, size=param_dict["viz_examples"], replace=False)
    for img_id in tqdm(img_ids_to_viz, desc="Vizualizing examples"):
        # read img and mask
        img = np.array(Image.open(img_dir / f"{img_id}.jpg"))
        mask = np.load(mask_dir / f"{img_id}.npy")
        # use augmentations
        transformed = augment(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]
        mask = torch.from_numpy(mask).to(torch.long)

        # transform input image for model type
        model_mask = segmentator.Segmentate(
            torch.unsqueeze(torch.from_numpy(np.moveaxis(img, 2, 0)).to(device, torch.float32), 0) 
        ).cpu()

        fig, ax = vizualizeSegmentation(
            img,
            model_mask.squeeze(0).numpy(),
            cl_to_name
        )
        ax.set_title(img_id)
        fig.savefig(segm_dir / f"{img_id}.jpg")

    # metrics containers
    mIoU_body = 0.0
    mIoU_up_low_body = 0.0
    mIoU_all = 0.0

    for i, img_id in tqdm(enumerate(img_ids), desc="Calculating metrics on validation"):
        # read img and mask
        img = np.array(Image.open(img_dir / f"{img_id}.jpg"))
        mask = np.load(mask_dir / f"{img_id}.npy")
        # use augmentations
        transformed = augment(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]
        mask = torch.from_numpy(mask).to(torch.long)

        # transform input image for model type
        model_mask = segmentator.Segmentate(
            torch.unsqueeze(torch.from_numpy(np.moveaxis(img, 2, 0)).to(device, torch.float32), 0) 
        ).cpu()

        # calaculate metrics

        # body
        mIoU_body += metrics.mIoU(
                    (model_mask != name_to_cl["bg"]).to(torch.long),
                    (mask != name_to_cl["bg"]).unsqueeze(0).to(torch.long),
                    [1],
                    "cpu"
        )[0].item()

        # low/up body
        transformed_model_mask = model_mask.detach()
        transformed_mask = mask.detach().unsqueeze(0)
        transformed_model_mask[
            torch.isin(
                model_mask,
                torch.LongTensor(
                    [name_to_cl["up_hand"], name_to_cl["low_hand"], name_to_cl["head"], name_to_cl["torso"]]
                )
            )
        ] = 1
        transformed_mask[
            torch.isin(
                mask,
                torch.LongTensor(
                    [name_to_cl["up_hand"], name_to_cl["low_hand"], name_to_cl["head"], name_to_cl["torso"]]
                )
            ).unsqueeze(0)
        ] = 1
        transformed_model_mask[
            torch.isin(
                model_mask,
                torch.LongTensor(
                    [name_to_cl["low_leg"], name_to_cl["up_leg"]]
                )
            )
        ] = 2
        transformed_mask[
            torch.isin(
                mask,
                torch.LongTensor(
                    [name_to_cl["low_leg"], name_to_cl["up_leg"]]
                )
            ).unsqueeze(0)
        ] = 2
        mIoU_up_low_body += metrics.mIoU(
                    transformed_model_mask,
                    transformed_mask,
                    [1, 2],
                    "cpu"
        )[0].item()

        # all classes except bg
        no_bg_classes = list(cl_to_name.keys())
        no_bg_classes.remove(name_to_cl["bg"])
        mIoU_all += metrics.mIoU(
                    model_mask,
                    mask.unsqueeze(0),
                    no_bg_classes,
                    "cpu"
        )[0].item()

    mIoU_body /= len(img_ids)
    mIoU_up_low_body /= len(img_ids)
    mIoU_all /= len(img_ids)

    metrics_table = pd.DataFrame.from_dict({
        "mIoU_body": [mIoU_body],
        "mIoU_up_low_body": [mIoU_up_low_body],
        "mIoU_all" : [mIoU_all]
    })
    metrics_table.to_csv(result_dir / "metrics/final_metrics.csv")
