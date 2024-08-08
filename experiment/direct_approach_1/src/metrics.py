import torch
import torch.nn as nn
import numpy as np

# батчированные

def mIoU(prediction: torch.Tensor, 
         target: torch.Tensor, 
         classes: list[int], 
         device: torch.device
) -> torch.Tensor:
    output = torch.zeros(prediction.shape[0], dtype=torch.float32).to(device)

    for cl in classes:
        pred_class_mask = (prediction == cl)
        targ_class_mask = (target == cl)

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
             device: torch.device, 
             bg_count: bool = False
) -> torch.Tensor:
    output = (prediction == target)
    # include background or not
    if not bg_count:
        output = output & (prediction != 0)

    return torch.mean(output.to(dtype=torch.float32), dim=(1, 2))


def gradNorm(model: nn.Module) -> float:
    with torch.no_grad():
        output = 0.
        for param in model.parameters():
            output += torch.sum(param.grad ** 2).item()

    return output ** 0.5





