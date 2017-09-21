import numpy as np

"""
    X: a mini-batch of N samples of shape(dim1, dim2, ..., dim k), so X is of shape(N, dim1, dim2, ..., dimk)
    W: Weight matrix of shape(dim1*dim2*dim3*..., M)
    b: shape (M, )
"""
def fc_forward(X, W, b):
    batch_size = X.shape[0]
    sampe_size = np.prod(X.shape[1:])
    _X = np.reshape(X(batch_size, sampe_size))
    output = np.dot(_X, W) + b
    cache = (X, W, b)

    return output, cache

def fc_backward(d_output, cache):
    X, W, b = cache
    d_X = np.dot(d_output, W.T).reshape(X.shape)
    d_W = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, d_output)
    d_b = np.sum(d_output, axis=0)

    return d_X, d_W, d_b

def relu_forward(X):
    output = np.maximum(0, X)
    cache = X

    return output, cache

def relu_backward(d_output, cache):
    X = cache
    d_X = np.array(d_output, copy=True)
    d_X[X <= 0] = 0

    return d_X

def bn_forward(X, gamma, beta, bn_param):
    """
    During training the sample mean and (uncorrected) sample variance are computed from mini-batch statistics,
    and used to normalise the incoming data.
    During training, we also keep an exponentially decaying running mean of the mean and variance of each feature,
    and these averages are used to normalised data at test-time.

    At each time step we update the running averages for mean and variance using an exponential decay based on the
    momentum parameter.

    :param X: (N, D)
    :param gamma: (D, )
    :param beta: (D, )
    :param bn_param: Dictionary with the following keys: mode, eps, momentum, running_mean, running_var
    :return:
    """
    mode = bn_param['mode']
    epsilon = bn_param.get('epsilon', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = X.shape

    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=X.dtype))
    running_variance = bn_param.get('running_variance', np.zeros(D, dtype=X.dtype))

    output, cache = None, None

    if mode == "train":
        X_mean = 1 / float(N) * np.sum(X, axis=0)
        X_white = X - X_mean
        X_pow = X_white ** 2
        var1 = 1 / float(N) * np.sum(X_pow, axis=0)
        sqrt_var = np.sqrt(var1 + epsilon)
        inv_var = 1. / sqrt_var
        var2 = X_white * inv_var
        var3 = gamma * var2
        out = var3 + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * X_mean
        running_variance = momentum * running_variance + (1.0 - momentum) * var1

        cache = (X_mean, X_white, X_pow, var1, sqrt_var, inv_var, var2, var3, gamma, beta, X, bn_param)

    elif mode == "test":
        X_mean = running_mean
        variance = running_variance
        X_hat = (X - X_mean) / np.sqrt(variance + epsilon)
        output = gamma * X_hat + beta
        cache = (X_mean, variance, gamma, beta, bn_param)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_variance

    return output, cache

def bn_backward(d_output, cache):
    d_X, d_gamma, d_beta = None, None, None
    X_mean, X_white, X_pow, var1, sqrt_var, inv_var, var2, var3, gamma, beta, X, bn_param = cache
    epsilon = bn_param.get('epsilon', 1e-5)
    N, D = d_output.shape

    d_var3 = d_output
    d_beta = np.sum(d_output, axis=0)

    d_var2 = gamma * d_var3
    d_gamma = np.sum(var2*d_var3, axis=0)

    d_X_white = inv_var * d_var2
    d_inv_var = np.sum(X_white*d_var2, axis=0)

    d_sqrt_var = -1./(sqrt_var**2)*d_inv_var

    d_var1 = 0.5*(var1+epsilon) ** (-0.5) * d_sqrt_var

    d_X_pow = 1/float(N) * np.ones((X_pow.shape)) * d_var1

    d_X_white += 2 * X_white * d_X_pow

    d_X = d_X_white
    d_X_mean = -np.sum(d_X_white, axis=0)

    d_X += 1/float(N) * np.ones((d_X_white.shape)) * d_X_mean

    return d_X, d_gamma, d_beta

def bn_backward_alt(d_output, cache):
    d_X, d_gamma, d_beta = None, None, None
    X_mean, X_white, X_pow, var1, sqrt_var, inv_var, var2, var3, gamma, beta, X, bn_param = cache
    epsilon = bn_param.get('epsilon', 1e-5)
    N, D = d_output.shape

    d_beta = np.sum(d_output, axis=0)
    d_gamma = np.sum((X - X_mean) * (var1 + epsilon)**(-1./2)* d_output, axis=0)
    d_X = (1./N)*gamma*(var1+epsilon)**(-1/2.)*(N*d_output - np.sum(d_output, axis=0) -
            (X - X_mean) * (var1+epsilon)**(-1.0) * np.sum(d_output*(X-X_mean), axis=0))

    return d_X, d_gamma, d_beta

def dropout_forward(X, dropout_param):
    ratio, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    output = None

    if mode == "train":
        mask = (np.random.random(*X.shape) < ratio)/ratio
        output = X * mask
    elif mode == "test":
        mask = None
        output = X

    cache = (dropout_param, mask)
    output = output.astype(X.dtype, copy=False)

    return output, cache

def dropout_backward(d_output, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']

    d_X = None

    if mode == "train":
        d_X = d_output * mask
    elif mode == "test":
        d_X = d_output

    return d_X

def conv_forward_naive(X, W, b, conv_param):
    output = None

    N, C, H, W = X.shape
    F, C, HH, WW = W.shape
    S = conv_param["stride"]
    P = conv_param["padding"]

    X_padding = np.pad(X, ((0, ), (0, ), (P, ), (P, )), 'constant')

    # Size of output
    Hh = 1 + (H + 2 * P - HH) / S
    Hw = 1 + (W + 2 * P - WW) / S

    output = np.zeros((N, F, Hh, Hw))

    for n in range(N):
        for f in range(F):
            for k in range(Hh):
                for l in range(Hw):
                    output[n, f, k, l] = np.sum(X_padding[n, :, k*S:k*S+HH, l*S:l*S+WW] * W[f, :]) + b[f]

    cache = (X, W, b, conv_param)

    return output, cache

def conv_backward_naive(d_output, cache):
    d_X, d_W, d_b = None, None, None

    X, W, b, conv_param = cache
    P = conv_param["padding"]
    X_padding = np.pad(X, ((0, ), (0, ), (P, ), (P, )), "constant")

    N, C, H, W = X.shape
    F, C, HH, WW = W.shape
    N, F, Hh, Hw = d_output.shape
    S = conv_param["stride"]

    d_W = np.zeros((F, C, HH, WW))
    dw = np.zeros((F, C, HH, WW))
    for f_prime in range(F):
        for c_prime in range(C):
            for i in range(HH):
                for j in range(WW):
                    sub_xpad = X_padding[:, c_prime, i:i + Hh * S:S, j:j + Hw * S:S]
                    dw[f_prime, c_prime, i, j] = np.sum(
                        d_output[:, f_prime, :, :] * sub_xpad)

    # For db : Size (F,)
    db = np.zeros((F))
    for f_prime in range(F):
        db[f_prime] = np.sum(d_output[:, f_prime, :, :])

    dx = np.zeros((N, C, H, W))
    for n_prime in range(N):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(Hh):
                        for l in range(Hw):
                            mask1 = np.zeros_like(W[f, :, :, :])
                            mask2 = np.zeros_like(W[f, :, :, :])
                            if (i + P - k * S) < HH and (i + P - k * S) >= 0:
                                mask1[:, i + P - k * S, :] = 1.0
                            if (j + P - l * S) < WW and (j + P - l * S) >= 0:
                                mask2[:, :, j + P - l * S] = 1.0
                            w_masked = np.sum(
                                W[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[n_prime, :, i, j] += d_output[n_prime, f, k, l] * w_masked

    return dx, dw, db

def maxpool_forward_naive(X, pool_param):
    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    S = pool_param['stride']
    N, C, H, W = X.shape
    H1 = (H - Hp) / S + 1
    W1 = (W - Wp) / S + 1

    output = np.zeros((N, C, H1, W1))
    for n in range(N):
        for c in range(C):
            for k in range(H1):
                for l in range(W1):
                    output[n, c, k, l] = np.max(
                        X[n, c, k * S:k * S + Hp, l * S:l * S + Wp])

    cache = (X, pool_param)
    return output, cache

def maxpool_backward_naive(d_output, cache):
    X, pool_param = cache
    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    S = pool_param['stride']
    N, C, H, W = X.shape
    H1 = (H - Hp) / S + 1
    W1 = (W - Wp) / S + 1

    d_X = np.zeros((N, C, H, W))
    for n_prime in range(N):
        for c_prime in range(C):
            for k in range(H1):
                for l in range(W1):
                    X_pooling = X[n_prime, c_prime, k * S:k * S + Hp, l * S:l * S + Wp]
                    maxi = np.max(X_pooling)
                    X_mask = X_pooling == maxi
                    d_X[n_prime, c_prime, k * S:k * S + Hp, l * S:l * S + Wp] += d_output[n_prime, c_prime, k, l]*X_mask
    return d_X


def spatial_bn_forrard(X, gamma, beta, bn_param):
    pass

def spatial_bn_backward(d_output, cache):
    pass

def svm_loss(X, y):

    N = X.shape[0]
    correct_class_scores = X[np.arange(N), y]
    margins = np.maximum(0, X - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    d_X = np.zeros_like(X)
    d_X[margins > 0] = 1
    d_X[np.arange(N), y] -= num_pos
    d_X /= N
    return loss, d_X

def softmax_loss(X, y):
    probabilities = np.exp(X - np.max(X, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    N = X.shape[0]
    loss = -np.sum(np.log(probabilities[np.arange(N), y])) / N
    d_X = probabilities.copy()
    d_X[np.arange(N), y] -= 1
    d_X /= N

    return loss, d_X