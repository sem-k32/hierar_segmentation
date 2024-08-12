import PIL.Image as Image
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
    ws_dir = pathlib.Path(os.environ["WORKSPACE_DIR"])
    with open(ws_dir / "Pascal-part/train_id.txt", "r") as f:
        train_ids = f.readlines()
        train_ids = list(map(lambda id: id.replace("\n", ""), train_ids))

    imgs_path = ws_dir / "Pascal-part/JPEGImages"
    mask_path = ws_dir / "Pascal-part/gt_masks"

    for train_id in train_ids:
        image = np.array(Image.open(imgs_path / f"{train_id}.jpg"))
        mask = np.load(mask_path / f"{train_id}.npy")

        if isGreyScale(image):
            grey_img_list.append(train_id)
        
        for i in range(param_dict["num_classes"]):
            mean_class_freq[i] += np.mean(mask == i)
    mean_class_freq /= len(train_ids)

    # main segmentation classes
    classes = {
        0: "bg",
        1: "low_hand",
        2: "torso",
        3: "low_leg",
        4: "head",
        5: "up_leg",
        6: "up_hand"
    }

    # save results
    result_dir = pathlib.Path("results/")
    result_dir.mkdir(exist_ok=True)
    results_dict = {
        "grey_img_ids": grey_img_list,
        "class_freq": mean_class_freq.tolist(),
        "classes": classes
    }
    with open(result_dir / "preprocess.yaml", "w") as f:
        yaml.dump(results_dict, f)


