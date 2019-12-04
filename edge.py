"""
    Edge detection obtained using learned CellNN templates.
"""

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
from util import converge, rgb2gray
import os

img = []
img_gray = []
target = []

train_dir = os.path.join(os.getcwd(), "edge_detection/training/images")
train_result_dir = os.path.join(os.getcwd(), "edge_detection/training/1st_manual")


for file in os.listdir(train_dir):
    print(file)
    file = os.path.join(train_dir, file)
    img.append(mpimg.imread(file))
    img_gray.append(rgb2gray(img[-1]))

for file in os.listdir(train_result_dir):
    print(file)
    file = os.path.join(train_result_dir, file)
    target.append(mpimg.imread(file))


def train():
    try:
        npzfile = np.load("edge.npz")
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
        new_rmses = []
        for p in range(len(img_gray)):
            ans = converge(img_gray[p], A, B, I, time=40, step=0.2)
            ans[ans == -1] = 0
            ans[ans == 1] = 255
            new_rmses.append(np.sqrt(((ans - target[p]) ** 2).sum() / ans.size))

        new_rmse = np.array(new_rmses).mean()
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
            np.savez("edge", A=A, B=B, I=I, rmses=rmses)
    np.savez("edge", A=A, B=B, I=I, rmses=rmses)
    return


def train_result():
    npzfile = np.load("edge.npz")
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


if __name__ == "__main__":
    # paper_result()
    # train_result()
    train()
