""" training first segmentation model to differentiate background and human
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import yaml
import pathlib
from tqdm import tqdm
from datetime import datetime

from src import metrics
from experiment.conseq_approach_1.model_1 import directSegmentator
from src.data_loader import prohibitBatchDataGetter, batchDataGetter
from params_1 import *


if __name__ == "__main__":
    # load params
    with open("params_1.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # load preprocess results
    with open("results/preprocess.yaml", "r") as f:
        preproc_dict = yaml.full_load(f)

    # device for train
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    # create model
    model = directSegmentator( 
        param_dict["model"]["num_levels"], 
        len(param_dict["classes"]),
        param_dict["model"]["kernal_size"],
        param_dict["model"]["num_conv_layers"],
        param_dict["model"]["encoder_dropout_p"],
        param_dict["model"]["leaky_relu_slope"]
    )
    model.to(device)

    # optimizer with l2 penalty, lr scheduler
    optimizer = optim.RMSprop(
        model.parameters(), 
        lr=param_dict["lr"],
        alpha=param_dict["grad_smooth"],
        momentum=param_dict["momentum"],
        weight_decay=param_dict["l2_pen"]
    )
    lr_sched = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda epoch: param_dict["lr"] if epoch < 750 else param_dict["lr"] / np.sqrt(epoch)
    )

    # functional to optimize with class weights and l2 penalty(set in optimizer)
    functional = nn.CrossEntropyLoss(
        weight=getClassesWeights().to(device)
    )

    # batched data loaders
    train_loader = getTrainDataLoader()
    valid_loader = getValDataLoader()

    # results dirs
    result_dir = pathlib.Path("results/")
    # metrics writer
    writer = SummaryWriter(result_dir / f"metrics/{param_dict['model']['name']}" / f"{datetime.now()}")
    # val images to vizaulize
    imgs_to_viz = getImgsToViz(param_dict["viz_examples"])

    # training cycle
    epoch_iter = tqdm(range(param_dict["max_epochs"]), desc="Loss: -")
    for epoch in epoch_iter:
        imgs, target_mask = train_loader.generateBatch()
        imgs = imgs.to(device)
        target_mask = target_mask.to(device)

        model.train()
        model_probs: torch.Tensor = model(imgs)

        optimizer.zero_grad()

        batch_loss = functional(model_probs, target_mask)
        batch_loss.backward()

        optimizer.step()
        lr_sched.step()

        with torch.no_grad():
            # log train metrics

            writer.add_scalar("Train/loss", batch_loss.item(), epoch)
            writer.add_scalar("Train/grad_norm", metrics.gradNorm(model), epoch)

            batch_mIoU = metrics.mIoU(
                model_probs.argmax(dim=1),
                target_mask,
                list(param_dict["classes"].keys()),
                device
            ).mean().item()
            writer.add_scalar("Train/mIoU", batch_mIoU, epoch)

            batch_accuracy = metrics.Accuracy(
                model_probs.argmax(dim=1),
                target_mask,
                list(param_dict["classes"].keys()),
                device
            ).mean().item()
            writer.add_scalar("Train/accuracy", batch_accuracy, epoch)

            if epoch % param_dict["validate_period"] == 0:
                # log validate metrics
                logValMetrics(epoch, model, device, functional, valid_loader, imgs_to_viz, writer)

                # backup model
                with open(result_dir / pathlib.Path(f"{param_dict['model']['name']}.pkl"), "wb") as f:
                    torch.save(model.state_dict(), f)

        # debug
        epoch_iter.set_description(f"Loss: {batch_loss.item()}")

    # last backup model
    with open(result_dir / pathlib.Path(f"{param_dict['model']['name']}.pkl"), "wb") as f:
        torch.save(model.state_dict(), f)

    # last log validate metrics
    with torch.no_grad():
        logValMetrics(param_dict["max_epochs"], model, device, functional, valid_loader, imgs_to_viz, writer)

    writer.close() 



