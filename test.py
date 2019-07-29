# %% read csv
import pandas as pd
import numpy as np
from Layer import *

df = pd.read_csv("./data/salary/train_sep.csv")
# %% prepare data
df_label = df.loc[:, 'income']
df_train = df.loc[:, :].drop('income', axis=1)

data_x = df_train.values.astype(np.float).T
data_y = df_label.values.astype(np.float).reshape((1, -1))

data_x = (data_x - np.mean(data_x, axis=1).reshape((-1, 1))) / np.std(data_x, axis=1).reshape((-1, 1))


# %%
def loss(pred, y):
    pred_log = np.log(pred + 1e-20)  # q(1) = f(x)
    no_pred_log = np.log(1.0 - pred + 1e-20)  # q(0) = 1 - f(x) ; f(x) : probability of class A(1)
    loss_val = -1. * (np.matmul(y, pred_log.T) + np.matmul((1.0 - y), no_pred_log.T))
    return loss_val

def grad_cross_entropy(pred,y):
    d_c_pred = -1.0 * (y / (pred + 1e-20) - (1. - y) / (1. - pred + 1e-20))  # (1,n) it is grad not loss
    return d_c_pred

dim, n = data_x.shape
dim1 = 64
dim2 = 16
dim3 = 4
dim4 = 2
dim5 = 1

layers = [Layer(dim,dim1,relu,relu_rev),
          Layer(dim1,dim2,relu,relu_rev),
          Layer(dim2,dim3,relu,relu_rev),
          #Layer(dim3,dim4,relu,relu_rev),
          Layer(dim3,dim5,sigmoid,sigmoid_rev)]

lr = 0.1
epochs = 2000
for epoch in range(epochs):
    a = data_x
    y = data_y

    # forward
    for layer in layers:
        a = layer.forward(a)

    # cross entropy
    d_c_a = grad_cross_entropy(a,y)  # (1,n) it is grad not loss

    # backpropagation
    for layer in reversed(layers):
        d_c_a = layer.backpropagation(d_c_a,lr)
    if epoch %1 == 0:
        loss_val = loss(a, y)[0][0]
        print("{:<5d},loss:{:.5f}".format(epoch,loss_val))