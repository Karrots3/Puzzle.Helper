from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def counter2mygrid(index, n_cols=10, start=0):
    """
    Convert a linear index to a grid position (row, col).
    """
    row = (index - start) // n_cols
    col = (index - start) % n_cols
    return row, col


def get_pieceId(piece_index: int, n_cols=10, start=0) -> str:
    """
    Generate a piece ID string based on the piece index.
    Args:
        piece_index (int): Index of the piece.
    Returns:
        str: Formatted piece ID.
    """
    row, col = counter2mygrid(piece_index, n_cols=n_cols, start=start)
    return f"{row:03d},{col}"


def plot_list_images(
    list_images,
    suptitle="Image Processing Pipeline",
    path_save: Path | None | str = None,
):
    """
    Plot a list of images in a grid layout with at most 4 columns.
    Args:
        list_images (dict): Dictionary with titles as keys and images as values.
        suptitle (str): Overall title for the figure.
        path_save (Path | None): Path to save the figure. If None, the figure is shown instead of saved.
    """
    num_images = len(list_images)
    cols = 3
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for i, (title, img) in enumerate(list_images.items()):
        if i < len(axes):
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap="gray")
            else:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(title, fontsize=12, fontweight="bold")
            axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(list_images), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(path_save, dpi=300, bbox_inches="tight")
    plt.close()
