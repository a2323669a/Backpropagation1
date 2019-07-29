# %% read csv
import pandas as pd
import numpy as np

df = pd.read_csv("./data/salary/train_ser.csv")
# %% prepare data
df_label = df.loc[:100, 'income']
df_train = df.loc[:100, :].drop('income', axis=1)

data_x = df_train.values.astype(np.float).T
data_y = df_label.values.astype(np.float).reshape((1, -1))


# %% train
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_inv(x):
    return sigmoid(x) * (1. - sigmoid(x))


def mul(x, w, b):
    return np.matmul(w.T, x) + b


def accuarcy(x, y, w, b):
    pred = f(x, w, b)
    pred = np.rint(pred)
    return np.mean(np.equal(pred, y).astype('float'))


dim1, n = data_x.shape
dim2 = 2
dim3 = 1
w1 = np.ones((dim1, dim2))  # 2 neurons
w2 = np.ones((dim2, dim3))  # 1 neurons output

b1 = np.zeros((1, 1))
b2 = np.zeros((1, 1))

w1_gss = np.zeros_like(w1)
w2_gss = np.zeros_like(w2)
b1_gss = np.zeros_like(b1)
b2_gss = np.zeros_like(b2)

lr = 1.
epochs = 100

for i in range(epochs):
    x = data_x
    y = data_y

    z1 = mul(x, w1, b1)  # (dim2,dim1) * (dim1,n) -> (dim2,n)
    a1 = sigmoid(z1)

    z2 = mul(a1, w2, b2)  # (dim3,dim2) * (dim2,n) -> (dim3,n)
    a2 = sigmoid(z2)

    d_c_a2 = -1.0 * (y / a2 - (1. - y) / (1. - a2))  # (1,n)

    d_w2 = np.sum(np.matmul(a1, np.multiply(sigmoid_inv(z2), d_c_a2)), axis=1)  # (2,n) * [(n,1) * (1,n)] -> (2,1)
    d_b2 = np.sum(np.matmul(np.ones((1, n)), np.multiply(sigmoid_inv(z2), d_c_a2)),
                  axis=1)  # (1,n) * [(n,1) * (1,n)] -> (1,1)
    d_c_a1 = np.matmul(w2, np.multiply(sigmoid_inv(z2), d_c_a2))  # (2,1) * (1,,n) * (1,n) -> (2,n)

    d_w1 = np.sum(np.matmul(x, np.multiply(sigmoid_inv(z1), d_c_a1)), axis=1)  # (dim,n) * [(n,2) * (2,n)] -> (dim,1)
    d_b1 = np.sum(np.matmul(np.ones((1, n)), np.multiply(sigmoid_inv(z1), d_c_a1)),axis=1)  # (1,n) * [(n,2) * (2,n)] -> (1,1)

    w1_gss += d_w1 ** 2
    w2_gss += d_w2 ** 2
    b1_gss += d_b1 ** 2
    b2_gss += d_b2 ** 2

    w1_t = w1 - (lr / np.sqrt(w1_gss)) * (d_w1)
    w2_t = w2 - (lr / np.sqrt(w2_gss)) * (d_w2)
    b1_t = b1 - (lr / np.sqrt(b1_gss)) * (d_b1)
    b2_t = b2 - (lr / np.sqrt(b2_gss)) * (d_b2)

    loss_val = np.sum(d_c_a2)[0]
    print("loss:{:.5f}", loss_val)
