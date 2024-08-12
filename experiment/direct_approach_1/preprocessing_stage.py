""" Find all grey images in train to further exclude them in train.
    Compute classes freqs.
    Read main classes for segmentation from data dir.
"""
import PIL.Image as Image
import numpy as np

import pathlib
import yaml
import os
import pickle


def isGreyScale(img: np.ndarray) -> bool:
    """
        Identifies if given image is grey-tone.
        img_dim = (h, w, c)

    """
    # move channel dim in the begining
    img = np.moveaxis(img, 2, 0)
    img[2] = img[2] - img[0]
    img[1] = img[1] - img[0]
    img[0] = img[0] - img[0]

    return np.all(img == 0)

    
if __name__ == "__main__":
    # load classes
    with open(os.environ["DATA_DIR"] + "/classes.pkl", "rb") as f:
        classes: dict = pickle.load(f)

    # containers for grey images and mean class frequences
    grey_img_list = []
    mean_class_freq = np.zeros(len(classes))

    # get train images id
    data_dir = pathlib.Path(os.environ["DATA_DIR"])
    with open(data_dir / "train_id.txt", "r") as f:
        train_ids = f.readlines()
        train_ids = list(map(lambda id: id.replace("\n", ""), train_ids))

    imgs_path = data_dir / "JPEGImages"
    mask_path = data_dir / "gt_masks"

    for train_id in train_ids:
        image = np.array(Image.open(imgs_path / f"{train_id}.jpg"))
        mask = np.load(mask_path / f"{train_id}.npy")

        if isGreyScale(image):
            grey_img_list.append(train_id)
        
        for i in classes.keys():
            mean_class_freq[i] += np.mean(mask == i)
    mean_class_freq /= len(train_ids)

    # save results
    result_dir = pathlib.Path("results/")
    result_dir.mkdir(exist_ok=True)
    results_dict = {
        "grey_img_ids": grey_img_list,
        "class_freq": mean_class_freq.tolist(),
    }
    with open(result_dir / "preprocess.yaml", "w") as f:
        yaml.dump(results_dict, f)


