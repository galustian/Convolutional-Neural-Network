import numpy as np

Weights = {}
Biases = {}


def add_padding_if_necessary(X, filter_size, stride):
    if (X.shape[0] - filter_size) % stride != 0:
        if X.shape[0] - filter_size < stride:
            remainder = stride - (X.shape[0] - filter_size)
        else:
            remainder = (X.shape[0] - filter_size) % stride
        remainder_l = int(remainder / 2)
        remainder_r = remainder - remainder_l
        X = np.pad(X, pad_width=((remainder_l, remainder_r), (0, 0), (0, 0)), mode='constant', constant_values=0)

    if (X.shape[1] - filter_size) % stride != 0:
        if X.shape[1] - filter_size < stride:
            remainder = stride - (X.shape[1] - filter_size)
        else:
            remainder = (X.shape[1] - filter_size) % stride
        remainder_l = int(remainder / 2)
        remainder_r = remainder - remainder_l
        X = np.pad(X, pad_width=((0, 0), (remainder_l, remainder_r), (0, 0)), mode='constant', constant_values=0)
    
    return X


def cnn_layer(X, layer_i, filter_layers=3, filter_size=3, stride=1, padding=0):
    if padding > 0:
        X = np.pad(X, pad_width=((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    
    X = add_padding_if_necessary(X, filter_size, stride)

    Z_col_size_h = int((X.shape[0] - filter_size) / stride + 1)
    Z_col_size_w = int((X.shape[1] - filter_size) / stride + 1)
    
    if layer_i not in Weights:
        Weights[layer_i] = np.random.randn(filter_size, filter_size, X.shape[2], filter_layers)
        Biases[layer_i] = np.zeros((Z_col_size_w, Z_col_size_h, filter_layers))

    V = np.zeros((Z_col_size_w, Z_col_size_h, filter_layers))

    for layer in range(V.shape[2]):
        for y in range(V.shape[1]):
            for x in range(V.shape[0]):
                V[x, y, layer] = np.sum(X[x*stride : x*stride+filter_size, y*stride : y*stride+filter_size,:] * Weights[layer_i][:,:,:,layer])

    V += Biases[layer_i]
    return V

def pool_layer(X, size, stride=None, mode='max'):
    if stride is None: stride = size

    X = add_padding_if_necessary(X, size, stride)

    h_size = int((X.shape[0] - size) / stride + 1)
    w_size = int((X.shape[1] - size) / stride + 1)

    V = np.zeros((h_size, w_size, X.shape[2]))

    for layer in range(V.shape[2]):
        for y in range(V.shape[1]):
            for x in range(V.shape[0]):
                X_block = X[x*stride : x*stride+size, y*stride : y*stride+size, layer]
                if mode == 'max':
                    V[x, y, layer] = np.max(X_block)
                elif mode == 'avg':
                    V[x, y, layer] = np.average(X_block)
    return V

def relu_layer(X):
    return np.maximum(0, X)


def softmax_cross_entropy_loss_layer(X, Y, layer_i, labels=2):
    X_flat = X.reshape(-1, 1)
    print(X)
    print(X_flat)

    if layer_i not in Weights:
        Weights[layer_i] = np.random.randn(labels, X_flat.shape[0])
        Biases[layer_i] = np.zeros((labels, 1))

    Z = np.dot(Weights[layer_i], X_flat)
    A_softmax = np.exp(Z) / np.sum(np.exp(Z))

    return A_softmax - Y


if __name__ == '__main__':
    v = cnn_layer(np.random.randn(8, 8, 3), 0, filter_layers=2, filter_size=3, stride=1)
    r = relu_layer(v)
    m = pool_layer(r, 2)
    v = cnn_layer(m, 1, filter_layers=3, filter_size=2, stride=1)
    r = relu_layer(v)

    Y = np.array([0, 1, 0, 0]).reshape(4, 1)
    Loss = softmax_cross_entropy_loss_layer(r, Y, 2, labels=4)
    print(Loss)