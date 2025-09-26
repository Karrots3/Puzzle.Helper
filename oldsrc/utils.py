import cv2
import numpy as np
import matplotlib.pyplot as plt



def imshow(img, title=None):
    plt.title(title)
    plt.axis('equal')
    plt.imshow(img)
    plt.show()

def plot_images(pair_images_title: list[tuple[np.ndarray, str]]):
    # plot images, if there are more than 4 then use a grid with 4 columns
    fig, axes = plt.subplots(1, len(pair_images_title), figsize=(10, 10))
    axes = axes.flatten()
    for i, (img, title) in enumerate(pair_images_title):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_contour(contour, **kwargs):
    plt.axis('equal')
    plt.plot(contour[:, :, 0], -contour[:, :, 1], **kwargs)

def plot_point(point, **kwargs):
    plot_contour(np.array([[point]]), **kwargs)