import matplotlib.pyplot as plt
import numpy as np
from save_list import load_list

def draw(y1, y2, name):
    plt.figure()
    x = np.arange(1, len(y1)+1)
    plt.title(name)
    if name == "Acc":
        plt.ylim(0.0, 1.0)
    else:
        plt.ylim(0, 3.0)
    plt.plot(x, y1, label="train")
    plt.plot(x, y2, label="val")
    plt.legend(loc='upper left')
    plt.savefig(name + ".png")


train_acc_list = load_list("train_acc")
val_acc_list = load_list("val_acc")
train_loss_list = load_list("train_loss")
val_loss_list = load_list("val_loss")
draw(train_acc_list, val_acc_list, "Acc")
draw(train_loss_list, val_loss_list, "Loss")
