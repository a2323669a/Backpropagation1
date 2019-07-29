import numpy as np

class Layer:

    def __init__(self,dim,neurons,f,f_rev):
        self.neurons = neurons
        self.dim = dim
        self.f = f
        self.f_rev = f_rev

        self.ws  = np.random.normal(0,0.02,(dim,neurons)) #(dim,neurons)
        #self.ws = np.ones((dim,neurons))
        self.bs  = np.zeros((neurons,1))

        self.ws_grad_square_sum = np.zeros_like(self.ws)
        self.bs_grad_square_sum = np.zeros_like(self.bs)

    def forward(self,x):
        self.x = x #(dim,n)
        self.z = np.matmul(self.ws.T,self.x) +  self.bs # (dim,neurons).T @ (dim,n) -> (neurons,n)
        self.a = self.f(self.z)
        return self.a

    def backpropagation(self,d_c_a,lr):
        # (dim,n) @  [(neurons,n) * (neurons,n)].T -> (dim,neurons)
        self.d_ws = np.matmul(self.x, np.multiply(self.f_rev(self.z), d_c_a).T)

        #(1,n) @ [(neurons,n) * (neurons,n)].T -> (1,neurons); (1,neurons).T -> (neurons,1)
        self.d_bs = np.matmul(np.ones((1,self.z.shape[1])),np.multiply(self.f_rev(self.z), d_c_a).T).T

        #(dim,neurons) @ [(neurons,n) * (neurons,n)] -> (dim,n)
        d_c_a0 = np.matmul(self.ws, np.multiply(self.f_rev(self.z), d_c_a))

        #update parameter
        self.ws_grad_square_sum += self.d_ws ** 2
        self.bs_grad_square_sum += self.d_bs ** 2

        ws_t = self.ws - (lr / np.sqrt(self.ws_grad_square_sum)) * (self.d_ws)
        bs_t = self.bs - (lr / np.sqrt(self.bs_grad_square_sum)) * (self.d_bs)

        self.ws = ws_t
        self.bs = bs_t

        return d_c_a0

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_rev(x):
    return sigmoid(x) * (1. - sigmoid(x))


def relu(x):
    return np.maximum(0, x)

def relu_rev(x):
    return np.where(x > 0, 1, 0)