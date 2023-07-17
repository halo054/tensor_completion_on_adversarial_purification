# mode-k unfolding of X (square bracket unfolding)
# Ik x Ik+1 ... IN I1...Ik-1

import numpy as np


def tenmat_sb(X, k):
    S = X.shape
    N = len(S)

    if k == 1:
        X_sb_k = np.reshape(X, (S[0], int(X.size / S[0])))
    elif k == N:
        X_sb_k = np.reshape(X, (int(X.size / S[N - 1]), S[N - 1]))
        X_sb_k = np.transpose(X_sb_k, (1, 0))

    else:
        X = np.reshape(X, (np.prod(S[0:k - 1]), int(X.size / np.prod(S[0:k - 1]))))
        X = np.transpose(X, [1, 0])
        X_sb_k = np.reshape(X, (S[k - 1], int(X.size / S[k - 1])))

    return X_sb_k


# formula 21 of <Tensor Ring Decomposition>
# Z is cell mode tensor cores
def TR_full_Z_k(Z, k):
    N = Z.size()


def Z_neq(Z, n):
    Z = np.roll(Z, -n - 1)  # arrange Z{n} to the last core, so we only need to multiply the first N-1 core
    N = np.size(Z, 0)
    P = Z[0]

    for i in range(N - 2):
        zl = np.reshape(P, (int(P.size / (np.size(Z[i], 2))), np.size(Z[i], 2)))
        zr = np.reshape(Z[i + 1], (np.size(Z[i + 1], 0), int(Z[i + 1].size / (np.size(Z[i + 1], 0)))))
        P = np.dot(zl, zr)
    Z_neq_out = np.reshape(P, (
    np.size(Z[0], 0), int(P.size / (np.size(Z[0], 0) * (np.size(Z[N - 1], 2)))), np.size(Z[N - 1], 2)))

    return Z_neq_out
