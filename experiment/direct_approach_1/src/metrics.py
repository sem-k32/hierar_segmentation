import torch
import torch.nn as nn
import numpy as np

# батчированные

def mIoU(prediction: torch.Tensor, target: torch.Tensor, classes: list[int]) -> torch.Tensor:
    output = torch.zeros(prediction.shape[0], dtype=torch.float32)

    for cl in classes:
        pred_class_mask = (prediction == cl)
        targ_class_mask = (target == cl)

        union_size = torch.sum(pred_class_mask | targ_class_mask, dim=(1, 2))
        intersection_size = torch.sum(pred_class_mask & targ_class_mask, dim=(1, 2))
        output += intersection_size / union_size

    output /= len(classes)

    return output


def Accuracy(prediction: torch.Tensor, target: torch.Tensor, bg_count: bool = False) -> torch.Tensor:
    output = (prediction == target)
    # include background or not
    if not bg_count:
        output = output & (prediction == 0)

    return torch.sum(output, dim=(1, 2))


def gradNorm(model: nn.Module) -> float:
    with torch.no_grad():
        output = 0.
        for param in model.state_dict().values():
            output += torch.sum(param ** 2).item()

    return output ** 0.5





