import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    h = sigmoid(X @ theta)
    cost = (-1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h)) + (lmd / (2 * m)) * (theta[1:] @ theta[1:])
    grad_unreg = (1 / m) * X.T @ (h - y)
    grad = grad_unreg.copy()
    grad[1:] = grad_unreg[1:] + (lmd / m) * theta[1:]

    # ===========================================================

    return cost, grad
