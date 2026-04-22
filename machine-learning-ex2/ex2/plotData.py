import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #
    pos = (y == 1)
    neg = (y == 0)
    plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=2, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markersize=7)

    # ===========================================================

