import torch
import torch.nn as nn
import numpy as np

# батчированные

def mIoU(prediction: torch.Tensor, 
         target: torch.Tensor, 
         classes: list[int], 
         device: torch.device,
         leave_bg: bool = False
) -> torch.Tensor:
    output = torch.zeros(prediction.shape[0], dtype=torch.float32).to(device)

    bg_mask = (target == 0)
    if leave_bg:
        classes.remove(0)

    for cl in classes:
        pred_class_mask = (prediction == cl)
        targ_class_mask = (target == cl)
        # discard bg pixels in prediction
        if leave_bg:
            pred_class_mask &= ~bg_mask

        # union may be zero
        union_size = torch.sum(pred_class_mask | targ_class_mask, dim=(1, 2)).to(dtype=torch.float32) + 1e-6
        intersection_size = torch.sum(pred_class_mask & targ_class_mask, dim=(1, 2)).to(dtype=torch.float32)
        # consider if current class exists on the picture
        output += (intersection_size / union_size) * torch.any(targ_class_mask, dim=(1, 2)) + \
            (~torch.any(targ_class_mask, dim=(1, 2))) * torch.ones(prediction.shape[0], dtype=torch.float32, device=device)

    output /= len(classes)

    return output


def Accuracy(prediction: torch.Tensor, 
             target: torch.Tensor,  
             classes: list[int],
             device: torch.device,
             leave_bg: bool = False
) -> torch.Tensor:
    if leave_bg:
        classes.remove(0)

    output = (prediction == target)
    # consider only given classes
    relevant_pixels = torch.isin(prediction, torch.LongTensor(classes).to(device))
    output = output & relevant_pixels

    return torch.sum(output, dim=(1, 2)) / torch.sum(relevant_pixels, dim=(1, 2))


def gradNorm(model: nn.Module) -> float:
    with torch.no_grad():
        output = 0.
        for param in model.parameters():
            output += torch.sum(param.grad ** 2).item()

    return output ** 0.5





