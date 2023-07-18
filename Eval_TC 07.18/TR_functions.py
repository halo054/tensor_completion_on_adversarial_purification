import numpy as np


def coreten2tr(Z):
    N = Z.size
    S = np.zeros(3, dtype=object)
    for i in range(N):
        S[i] = Z[i].shape[1]

    P = Z[0]
    for i in range(1, N):
        L = np.reshape(P, (int(P.size / Z[i - 1].shape[2]), Z[i - 1].shape[2]))
        R = np.reshape(Z[i], (Z[i].shape[0], int(S[i] * Z[i].shape[2])))

        P = np.dot(L, R)

    P = np.reshape(P, (Z[0].shape[0], np.prod(S), Z[N - 1].shape[2]))
    P = np.transpose(P, [1, 2, 0])
    P = np.reshape(P, (np.prod(S), int(Z[0].shape[0] * Z[0].shape[0])))

    temp = np.eye(Z[0].shape[0], Z[0].shape[0])

    P = np.dot(P, (temp.T.flatten()))
    X = np.reshape(P, S)

    return X


# generate tensor W by size and missing rate
def gen_W(S, mr):
    # np.prod(S)  prod：所有元素的乘积，即X三维乘积

    Omega_1 = np.random.permutation(np.prod(S))  # 生成长度为 S_prod 的序列并打乱顺序
    Omega = Omega_1[0:int(round((1 - mr) * np.prod(S)))].tolist()
    W_1 = np.zeros(np.prod(S))

    W_1[Omega] = 1
    W = W_1.reshape(S)

    return W


# fold core tensor to original size
def Gfold(Gm, SGt, n):
    if n == 0:
        Gt_out = np.reshape(Gm, (SGt[0], SGt[1], SGt[2]))
    elif n == 1:
        Gt_out = np.transpose(np.reshape(Gm, (SGt[1], SGt[0], SGt[2])), (1, 0, 2))
    elif n == 2:
        Gt_out = np.transpose(np.reshape(Gm, (SGt[2], SGt[0], SGt[1])), (1, 2, 0))
    else:
        print('wrong!')

    return Gt_out


# only for unfold core tensors to 3 different mode
def Gunfold(Gt, n):
    if n == 0:
        Gm = np.reshape(Gt, (Gt.shape[0], Gt.shape[1] * Gt.shape[2]))
    elif n == 1:
        Gm = np.reshape(np.transpose(Gt, (1, 0, 2)), (Gt.shape[1], Gt.shape[0] * Gt.shape[2]))
    elif n == 2:
        Gm = np.reshape(np.transpose(Gt, (2, 0, 1)), (Gt.shape[2], Gt.shape[0] * Gt.shape[1]))
    else:
        print('wrong!')
    return Gm


def Msum_fun(M):
    N = np.size(M, 0)
    Msum_out = np.zeros((N,), dtype=np.object)
    for i in range(N):
        Msum_out[i] = M[i][0] + M[i][1] + M[i][2]

    return Msum_out


# from tensor to matrix
# 1 kolda mat, 2 tt mat, 3 tr mat
def mytenmat(X, n, type):
    S = X.shape
    N = len(S)
    if type == 1:  # the In x I1...In-1In+1...IN type
        if n == 1:
            Xn = np.reshape(X, (S[0], np.prod(S[1:])))
        elif n == N:
            Xn = np.transpose(np.reshape(X, (np.prod(S[0:-1]), S[N - 1])), (1, 0))
        else:
            arr_1 = np.array(n - 1)
            arr_2 = np.arange(0, n - 1)
            arr_3 = np.arange(n, N)
            arr = np.append(np.append(arr_1, arr_2), arr_3)
            Xn = np.reshape(np.transpose(X, arr), (S[n - 1], int(np.prod(S) / S[n - 1])))

    elif type == 2:  # the I1...In x In+1...IN type
        Xn = np.reshape(X, (np.prod(S[0:n]), int(X.size / np.prod(S[0:n]))))

    else:  # the In x In+1...INI1...In-1 type
        if n == 1:
            Xn = np.reshape(X, (S[0], int(X.size / S[0])))
        elif n == N:
            Xn = np.reshape(X, (int(X.size / S[N - 1]), S[N - 1]))
            Xn = np.transpose(Xn, (1, 0))
        else:
            Xn = np.reshape(X, (np.prod(S[0:n - 1]), int(X.size / np.prod(S[0:n - 1]))))
            Xn = np.transpose(Xn, (1, 0))
            Xn = np.reshape(Xn, (S[n - 1], int(X.size / S[n - 1])))

    return Xn


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min: 1/2*||Z-X||^2 + ||X||_tr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% [S, V, D, Sigma2] = MySVDtau(Z, tau);
% V = max(diag(V) - tau, 0);
% n = sum(V > 0);
% X = S(:, 1:n) * diag(V(1:n)) * D(:, 1:n)';

%% new
'''

def get_max_value(arr):
    m, n = np.shape(arr)
    max_matlab = np.zeros(n, dtype=type(arr[1][1]))
    for i in range(n):
        max_col = np.max(arr[:, i])
        max_matlab[i] = max_col
    return max_matlab


def Pro2TraceNorm(Z, tau):
    (m, n) = Z.shape

    if 2 * m < n:
        AAT = np.dot(Z, Z.T)
        S, Sigma2, D = np.linalg.svd(AAT)

        V = np.sqrt(Sigma2)

        tol = max(Z.shape) * np.finfo(max(V)).eps
        n = sum(V > max(tol, tau))

        mid = np.maximum(V[0:n - 1] - tau, 0) / V[0: n - 1]

        X = np.dot(np.dot(np.dot(S[:, 0:n - 1], np.diag(mid)), S[:, 0:n - 1].T), Z)

    elif m > 2 * n:
        X, n, Sigma2 = Pro2TraceNorm(Z.T, tau)
        X = X.T
    else:
        S, V, D = np.linalg.svd(Z,full_matrices=False)
        Sigma2 = V ** 2
        n = sum(V > tau)
        S1 = S[:, 0:n - 1]
        V = np.diag(V)
        V1 = np.maximum(V[0:n - 1,0:n - 1] - tau, 0)
        temp = np.dot(S1,V1)
        X = np.dot(temp, (D[ :,0:n - 1].T))

    return X, n, Sigma2


def RSE_fun(X, X_hat, W):
    # global RSE
    RSE_list = [0, 0, 0]
    err = X_hat.T.flatten() - X.T.flatten()
    RSE_list[0] = np.sqrt(np.sum(err ** 2) / np.sum((X.T.flatten()) ** 2))

    # RSE of observed entries
    X_hat_w = X_hat * W
    X_w = X * W
    err_w = X_hat_w.T.flatten() - X_w.T.flatten()
    RSE_list[1] = np.sqrt(np.sum(err_w ** 2) / np.sum((X_w.T.flatten()) ** 2))

    # RSE of missing entries
    Wr = np.logical_not(W)
    X_hat_wr = X_hat * Wr
    X_wr = X * Wr
    err_wr = X_hat_wr.T.flatten() - X_wr.T.flatten()
    RSE_list[2] = np.sqrt(np.sum(err_wr ** 2) / np.sum((X_wr.T.flatten()) ** 2))

    return RSE_list


# this is for simulation data, randomly generate tensor cores Z
def TR_initcoreten(S, r):
    N = int(np.size(S))
    Z = np.zeros((3,), dtype=np.object)  # 设置 dtype=np.object，可以在矩阵中设置形状不同的子矩阵
    for i in range(N - 1):
        Z[i] = np.random.randn(r[i], S[i], r[i + 1])
    Z[N - 1] = np.random.randn(r[N - 1], S[N - 1], r[0])

    return Z
