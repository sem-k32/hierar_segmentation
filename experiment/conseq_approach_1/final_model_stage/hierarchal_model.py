import torch
import torch.nn as nn

import os
import pickle
from copy import deepcopy

class hierarBodySegmentator:
    def __init__(
            self,
            models_dict: dict[str, nn.Module],
            model_class_to_name_dict: dict[str, dict]
    ) -> None:
        self._models = models_dict
        # switch model to eval mode
        for model in models_dict.values():
            model.eval()

        self._model_cl_to_name = model_class_to_name_dict
        # make reverse mappings
        self._model_name_to_cl = {}
        for model_name, cl_to_name_dict in model_class_to_name_dict.items():
            self._model_name_to_cl[model_name] = {
                name: cl for cl, name in cl_to_name_dict.items()
            }

        # load inital classes
        with open(os.environ["DATA_DIR"] + "/classes.pkl", "rb") as f:
            self.cl_to_name: dict = pickle.load(f)
            # make reverse mappings
            self._name_to_cl = {
                name: cl for cl, name in self.cl_to_name.items()
            }

    # input is batched
    def Segmentate(self, imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # separate on bg and humans
            bg_human_mask: torch.Tensor = self._models["bg_human"](imgs).argmax(dim=1)

            # separate on lower/upper body
            human_cl = self._model_name_to_cl["bg_human"]["human"]
            transformed_imgs = imgs.detach()
            transformed_imgs[bg_human_mask.unsqueeze(1).repeat(1, 3, 1, 1) != human_cl] = 0.0
            low_up_body_mask = self._models["low_up_body"](transformed_imgs).argmax(dim=1) + 1

            # separate upper body classes
            up_body_cl = self._model_name_to_cl["low_up_body"]["upper_body"]
            transformed_imgs = imgs.detach()
            transformed_imgs[low_up_body_mask.unsqueeze(1).repeat(1, 3, 1, 1) != up_body_cl] = 0.0
            upper_body_mask = self._models["up_body"](transformed_imgs).argmax(dim=1) + 1

            # separate lower body classes
            low_body_cl = self._model_name_to_cl["low_up_body"]["lower_body"]
            transformed_imgs = imgs.detach()
            transformed_imgs[low_up_body_mask.unsqueeze(1).repeat(1, 3, 1, 1) != low_body_cl] = 0.0
            lower_body_mask = self._models["low_body"](transformed_imgs).argmax(dim=1) + 1

            # consequently build segmentation mask
            final_mask = torch.empty_like(bg_human_mask)

            # bg
            final_mask[bg_human_mask != human_cl] = self._name_to_cl["bg"]
            # low leg
            final_mask[
                (bg_human_mask == human_cl) & 
                (low_up_body_mask == self._model_name_to_cl["low_up_body"]["lower_body"]) & 
                (lower_body_mask == self._model_name_to_cl["low_body"]["low_leg"])
            ] = self._name_to_cl["low_leg"]
            # up leg
            final_mask[
                (bg_human_mask == human_cl) & 
                (low_up_body_mask == self._model_name_to_cl["low_up_body"]["lower_body"]) & 
                (lower_body_mask == self._model_name_to_cl["low_body"]["up_leg"])
            ] = self._name_to_cl["up_leg"]
            temp = (final_mask == self._name_to_cl["up_leg"]).sum()
            # low hand
            final_mask[
                (bg_human_mask == human_cl) & 
                (low_up_body_mask == self._model_name_to_cl["low_up_body"]["upper_body"]) & 
                (upper_body_mask == self._model_name_to_cl["up_body"]["low_hand"])
            ] = self._name_to_cl["low_hand"]
            # up hand
            final_mask[
                (bg_human_mask == human_cl) & 
                (low_up_body_mask == self._model_name_to_cl["low_up_body"]["upper_body"]) & 
                (upper_body_mask == self._model_name_to_cl["up_body"]["up_hand"])
            ] = self._name_to_cl["up_hand"]
            # torso
            final_mask[
                (bg_human_mask == human_cl) & 
                (low_up_body_mask == self._model_name_to_cl["low_up_body"]["upper_body"]) & 
                (upper_body_mask == self._model_name_to_cl["up_body"]["torso"])
            ] = self._name_to_cl["torso"]
            # head
            final_mask[
                (bg_human_mask == human_cl) & 
                (low_up_body_mask == self._model_name_to_cl["low_up_body"]["upper_body"]) & 
                (upper_body_mask == self._model_name_to_cl["up_body"]["head"])
            ] = self._name_to_cl["head"]

        return final_mask


            

