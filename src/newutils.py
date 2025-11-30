import matplotlib.pyplot as plt
import numpy as np


def plt_np_3ch(
    plots: list[np.ndarray],
    size_height: int = 3,
    max_col: int = 5,
    ibw: int | list[int] | range | None = None,
):
    if ibw is None:
        ibw = []
    elif isinstance(ibw, range):
        ibw = list(ibw)
    elif isinstance(ibw, int):
        ibw = [ibw]

    height, width = plots[0].shape[:2]
    ratio = width / height

    n_plots = len(plots)
    n_col = min(n_plots, max_col)
    n_row = int(np.ceil(n_plots / n_col))

    fig_size = (size_height * ratio * n_col, size_height * n_row)

    _, axes = plt.subplots(n_row, n_col, figsize=fig_size)
    axes = axes.flatten() if n_plots > 1 else [axes]
    for i, ax in enumerate(axes):
        is_bw = False
        if i in ibw:
            is_bw = True

        if i < n_plots:
            ax.imshow(plots[i], cmap="gray" if is_bw else None)
            ax.axis("off")
        else:
            ax.remove()
    plt.tight_layout()
    plt.show()
