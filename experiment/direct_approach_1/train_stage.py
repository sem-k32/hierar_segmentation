""" training model to predict all classes at once
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from torch.utils.tensorboard import SummaryWriter
import yaml
import pathlib
import gc
from tqdm import tqdm
from datetime import datetime

from src import metrics
from src.model import directSegmentator
from src.data_loader import getTrainBatchLoader, testDataIter
from src.vizualize import vizualizeSegmentation


def getClassesWeights(preprocess_res: dict, params_dict: dict) -> torch.Tensor:
    """strategy for class weightening
    """
    # non-bg classes are equally important (importance = 1)
    class_weights = torch.ones(len(preprocess_res["class_freq"]), dtype=torch.float32)
    # bg class is some times less important
    class_weights[0] = 1 / 10

    return class_weights

def logValidateMetrics(
        epoch: int,
        model: nn.Module,
        device: torch.device,
        functional,
        valid_loader,
        writer: SummaryWriter
) -> None:
    model.eval()

    # log validate metrics
    val_loss = 0
    val_mIoU = 0
    val_accuracy = 0
    num_batches = 0

    for val_img, val_masks in tqdm(valid_loader, desc="Validation", leave=False):
        val_img = val_img.to(device)
        val_masks = val_masks.to(device)

        val_probs: torch.Tensor = model(val_img)

        val_loss += functional(val_probs, val_masks).item()
        val_mIoU += metrics.mIoU(
                        val_probs.argmax(dim=1),
                        val_masks,
                        list(range(1, param_dict["num_classes"])),
                        device
                    ).mean().item()
        val_accuracy += metrics.Accuracy(
                            val_probs.argmax(dim=1),
                            val_masks,
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
        imgs, _ = next(iter(valid_loader))
        model_mask = model(imgs.detach().to(device)).argmax(dim=1)
    for i in range(param_dict["viz_examples"]):
        fig, ax = vizualizeSegmentation(
            np.moveaxis(imgs[i].numpy(), 0, 2).astype(np.int32),
            model_mask[i].cpu().numpy(),
            param_dict["classes"]
        )
        ax.set_title(f"Test example {i}")

        writer.add_figure(f"Test/segmentation/expl_{i}", fig, epoch)


if __name__ == "__main__":
    # load params
    with open("params.yaml", "r") as f:
        param_dict = yaml.full_load(f)
    # load preprocess results
    with open("results/preprocess.yaml", "r") as f:
        preprocess_res = yaml.full_load(f)

    # device for train
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    # create model
    model = directSegmentator(param_dict["model"]["num_levels"], num_classes=param_dict["num_classes"])
    model.to(device)

    # optimizer with l2 penalty, lr scheduler
    optimizer = optim.RMSprop(
        model.parameters(), 
        lr=param_dict["lr"],
        alpha=param_dict["grad_smooth"],
        momentum=param_dict["momentum"],
        weight_decay=param_dict["l2_pen"]
    )
    lr_sched = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: param_dict["lr"])

    # functional to optimize with class weights and l2 penalty(set in optimizer)
    functional = nn.CrossEntropyLoss(
        weight=getClassesWeights(preprocess_res, param_dict).to(device)
    )

    # batched data loaders
    train_loader = getTrainBatchLoader(
        param_dict["batch_size"],
        prohibit_images_id=preprocess_res["grey_img_ids"]
    )()
    valid_loader = testDataIter(param_dict["batch_size"])

    # results dirs
    result_dir = pathlib.Path("results/")
    # metrics writer
    writer = SummaryWriter(result_dir / f"metrics/" / f"{datetime.now()}")

    # training cycle
    epoch_iter = tqdm(range(param_dict["max_epochs"]), desc="Loss: -")
    for epoch in epoch_iter:
        imgs, target_mask = next(train_loader)
        imgs = imgs.to(device)
        target_mask = target_mask.to(device)

        model.train()
        model_probs: torch.Tensor = model(imgs)

        # try save cuda memory
        del imgs
        gc.collect()

        optimizer.zero_grad()

        batch_loss = functional(model_probs, target_mask)
        batch_loss.backward()

        optimizer.step()
        lr_sched.step()

        with torch.no_grad():
            # log train metrics
            writer.add_scalar("Train/loss", batch_loss.item(), epoch)
            writer.add_scalar("Train/grad_norm", metrics.gradNorm(model), epoch)
            # do not count bg-class
            batch_mIoU = metrics.mIoU(
                model_probs.argmax(dim=1),
                target_mask,
                list(range(1, param_dict["num_classes"])),
                device
            ).mean().item()
            writer.add_scalar("Train/mIoU", batch_mIoU, epoch)
            # do not count bg-class
            batch_accuracy = metrics.Accuracy(
                model_probs.argmax(dim=1),
                target_mask,
                device
            ).mean().item()
            writer.add_scalar("Train/accuracy", batch_accuracy, epoch)

            if epoch % param_dict["validate_period"] == 0:
                # log validate metrics
                logValidateMetrics(epoch, model, device, functional, valid_loader, writer)

                # backup model
                with open(result_dir / pathlib.Path("model.pkl"), "wb") as f:
                    torch.save(model.state_dict(), f)

        # debug
        epoch_iter.set_description(f"Loss: {batch_loss.item()}")

    # last backup model
    with open(result_dir / pathlib.Path("model.pkl"), "wb") as f:
        torch.save(model.state_dict(), f)

    # last log validate metrics
    with torch.no_grad():
        logValidateMetrics(param_dict["max_epochs"], model, device, functional, valid_loader, writer)

    writer.close() 



