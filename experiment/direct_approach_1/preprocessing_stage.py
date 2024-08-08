import PIL
import numpy as np

import pathlib
import yaml
import os


def isGreyScale(img: np.ndarray) -> bool:
    # make channel dim first
    img = np.moveaxis(img, 2, 0)
    img[2] = img[2] - img[0]
    img[1] = img[1] - img[0]
    img[0] = img[0] - img[0]

    return np.all(img == 0)
    

""" find all grey images in train to remove them
    count classes freqs to make class weightening in train
"""
if __name__ == "__main__":
    # load params
    with open("params.yaml", "r") as f:
        param_dict = yaml.full_load(f)

    # containers for grey images and mean class frequences
    grey_img_list = []
    mean_class_freq = np.zeros(param_dict["num_classes"])

    # get train images id
    ws_dir = pathlib.Path(os["WORKSPACE_DIR"])
    with open(ws_dir / "Pascal-part/train_id.txt", "r") as f:
        train_ids = f.readlines()

    imgs_path = pathlib.Path(ws_dir / pathlib.Path("/Pascal-part/JPEGImages"))
    mask_path = pathlib.Path(ws_dir / pathlib.Path("/Pascal-part/gt_masks"))

    for train_id in train_ids:
        image = np.array(PIL.Image.open(imgs_path / pathlib.Path(train_id + ".jpg")))
        mask = np.load(mask_path / pathlib.Path(train_id + ".npy"))

        if isGreyScale(image):
            grey_img_list.append(train_id)
        
        for i in range(param_dict["num_classes"]):
            mean_class_freq[i] += np.mean(mask == i)
    mean_class_freq /= len(train_ids)
    # transform to class weights
    class_weights = 1 / mean_class_freq

    # save results
    result_dir = pathlib.Path("results/").mkdir(exist_ok=True)
    results_dict = {
        "grey_img_ids": grey_img_list,
        "class_weights": class_weights.tolist()
    }
    with open(result_dir / pathlib.Path("preprocess.yaml"), "w") as f:
        yaml.dump(results_dict, f)


