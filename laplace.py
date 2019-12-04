"""
    Laplacian obtained using learned CellNN templates.
"""

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
from scipy import ndimage
from util import converge, rgb2gray

# img = mpimg.imread("lena_std.tif")
img = mpimg.imread("baboon.jpg")
img_gray = rgb2gray(img)

laplace_template = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

target = ndimage.convolve(img_gray / np.abs(img_gray).max(), laplace_template)
target = target / (np.abs(target).max())


def train():
    try:
        npzfile = np.load("laplace.npz")
        rmses = npzfile["rmses"].tolist()
        A = npzfile["A"]
        B = npzfile["B"]
        I = npzfile["I"]
    except Exception:
        rmses = []
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        I = 0

    delta = 0.0001
    dw_A = delta * np.random.choice([-1, 1], (3, 3))
    dw_A[1][2] = dw_A[1][0]
    dw_A[2][0] = dw_A[0][2]
    dw_A[2][1] = dw_A[0][1]
    dw_A[2][2] = dw_A[0][0]
    dw_B = delta * np.random.choice([-1, 1], (3, 3))
    dw_I = delta * np.random.choice([-1, 1])
    rmse = 100000000
    for i in range(100000):
        A = A + dw_A
        B = B + dw_B
        I = I + dw_I
        ans = converge(img_gray / np.abs(img_gray).max(), A, B, I, time=40, step=0.2)
        new_rmse = np.sqrt(((ans - target) ** 2).sum() / ans.size)
        if new_rmse > rmse:
            dw_A = delta * np.random.choice([-1, 1], (3, 3))
            dw_A[1][2] = dw_A[1][0]
            dw_A[2][0] = dw_A[0][2]
            dw_A[2][1] = dw_A[0][1]
            dw_A[2][2] = dw_A[0][0]
            dw_B = delta * np.random.choice([-1, 1], (3, 3))
            dw_I = delta * np.random.choice([-1, 1])
        rmse = new_rmse
        rmses.append(rmse)
        print("epoch= %s RMSE = %s" % (i, rmse))
        if i % 100 == 0:
            np.savez("laplace", A=A, B=B, I=I, rmses=rmses)
    np.savez("laplace", A=A, B=B, I=I, rmses=rmses)
    return


def train_result():
    npzfile = np.load("laplace.npz")
    rmses = npzfile["rmses"]
    A = npzfile["A"]
    B = npzfile["B"]
    I = npzfile["I"]

    ans = converge(img_gray / np.abs(img_gray).max(), A, B, I, time=40, step=0.2)

    rmse = np.sqrt(((ans - target) ** 2).sum() / ans.size)

    plt.figure()
    f1 = plt.subplot(2, 2, 1)
    f2 = plt.subplot(2, 2, 2)
    f3 = plt.subplot(2, 2, 3)
    f4 = plt.subplot(2, 2, 4)

    target[target < 0] = 0
    target[target > 1] = 1
    ans[ans < 0] = 0
    ans[ans > 1] = 1

    f1.axis("off")
    f2.axis("off")
    f3.axis("off")

    f1.set_title("Original")
    f1.imshow(img_gray, cmap="gray")

    f2.set_title("CellNN after training\n RMSE = %s" % (rmse))
    f2.imshow(ans, cmap="gray")

    f3.set_title("Laplace template")
    f3.imshow(target, cmap="gray")

    f4.set_title("Training error vs epoch")
    f4.set_xlabel("Epochs")
    f4.set_ylabel("RMSE")
    f4.plot(np.arange(rmses.size), rmses)
    plt.tight_layout()
    plt.show()
    return


def paper_result():
    A = np.array(
        [[0.335, 0.176, -0.535], [-0.298, -2.761, -0.298], [-0.535, 0.176, 0.335]]
    )
    B = np.array(
        [[0.174, -0.441, -0.853], [-0.576, 3.992, -0.192], [-1.231, 0.0, -0.810]]
    )
    I = -0.034

    ans = converge(img_gray / np.abs(img_gray).max(), A, B, I, time=40, step=0.2)

    rmse = np.sqrt(((ans - target) ** 2).sum() / ans.size)
    print("RMSE = ", rmse)

    plt.figure(figsize=(15, 6))
    f1 = plt.subplot(1, 3, 1)
    f2 = plt.subplot(1, 3, 2)
    f3 = plt.subplot(1, 3, 3)

    f1.axis("off")
    f2.axis("off")
    f3.axis("off")

    f1.set_title("Original")
    f1.imshow(img_gray, cmap="gray")

    f2.set_title("CellNN in the paper\n" + "RMSE =  %.3f" % rmse)
    f2.imshow(ans, cmap="gray")

    f3.set_title("Laplace template")
    f3.imshow(target, cmap="gray")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    paper_result()
    train_result()
    # train()
