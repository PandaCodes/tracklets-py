from matplotlib import pyplot as plt
import numpy as np


def imshow2(img1, img2):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img2)
    plt.show()


def imshow_all(*imgs, **kwargs):
    hide_axis = True if "hide_axis" not in kwargs else kwargs["hide_axis"]
    fig = plt.figure()
    for i in range(len(imgs)):
        ax = fig.add_subplot(1, len(imgs), i + 1)
        if hide_axis:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        ax.imshow(imgs[i], **kwargs)
    plt.show()


def imshow_surface(img, rotate=(0, 0)):
    X = np.linspace(0, img.size(0), img.size(0))
    Y = np.linspace(0, img.size(1), img.size(1))
    X, Y = np.meshgrid(X, Y, indexing='ij')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # TODO: show max contur
    ax.plot_surface(X, Y, img, cmap="plasma")
    ax.view_init(*rotate)

    plt.show()
