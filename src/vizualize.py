import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


def vizualizeSegmentation(img: np.ndarray, segm_mask: np.ndarray, classes: dict):
    """vizualize segmentation over given image, consider only given classes

    Args:
        classes (dict): classes to consider for vizualization

    Returns:
        mpl.Figure, mpl.Axes
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)

    # make colorbar
    cmap = mpl.cm.gist_ncar
    bounds = list(range(len(classes) + 1))
    norm =  mpl.colors.BoundaryNorm(bounds, cmap.N, extend="both")
    # draw mask
    ax.imshow(
        segm_mask,
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )
    # draw colorbar
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, orientation='vertical'
    )
    colorbar.set_ticklabels(list(classes.values()) + [" "])

    return fig, ax

