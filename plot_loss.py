
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(loss_arr):
    loss_ar = np.array(loss_arr)

    x = np.arange(1, 21)
    y = loss_ar
    
    # plotting
    from matplotlib.pyplot import figure

    fig = figure(figsize=(8, 6), dpi=80)
    plt.title("NER Loss function")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.plot(x, y, color ="red")
    plt.show()
    fig.savefig('figures/loss.pdf')