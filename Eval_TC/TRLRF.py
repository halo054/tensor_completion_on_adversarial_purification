'''
------------------------------------------------------------------------------------------------
% ALGORITHM:
% Tensor-ring low-rank factors (TRLRF)
% Time: 12/25/2018
% Reference: "Tensor Ring Decomposition with Rank Minimization on Latent Space:
%                  An Efficient Approach for Tensor Completion", AAAI, 2019.
------------------------------------------------------------------------------------------------
% MODEL:
% \min \limits_{[\tensor{G}],\tensor{X}}  \ &\sum_{n=1}^N\sum_{i=1}^3 \Vert \mat{G}^{(n)}_{(i)} \Vert_*
% + \frac{\lambda}{2}\Vert \tensor{X}-\Psi([\tensor{G}])\Vert_F^2\\&
% s.t.\ P_\Omega(\tensor{X})=P_\Omega(\tensor{T}).
------------------------------------------------------------------------------------------------
% INPUT
% data: incomplete tensor
% W: binary tensor, 1 means observed entries, 0 means missing entries
% r: TR-rank
% maxiter: 300~500
% K: hyper-parameter 0.1~1
% ro: hyper-parameter 1~1.5
% Lambda: hyper-parameter 1~10
% tol: if \Vert \tensor{X}-\tensor{X}_{last} \Vert_F / \Vert \tensor{X}\Vert_F<tol; break
------------------------------------------------------------------------------------------------
% OUTPUT
% X: completed tensor
% G_out: factors of TR decomposition
% Convergence_rec: records of loss function values
------------------------------------------------------------------------------------------------
'''

import numpy as np
from TR_functions import TR_initcoreten, Msum_fun, Gfold, Gunfold, Pro2TraceNorm, coreten2tr, mytenmat
from TRWOPT_functions import Z_neq, tenmat_sb


def TRLRF(data, W, r, maxiter, K, ro, Lambda, tol):
    T = data * W
    N = T.ndim
    S = T.shape
    S_1, S_2, S_3 = S
    X = np.random.rand(S_1, S_2, S_3)

    G = TR_initcoreten(S, r)

    M = np.zeros((N, 3), dtype=np.object)
    Y = np.zeros((N, 3), dtype=np.object)

    for i in range(N):
        G[i] = 1 * G[i]
        for j in range(3):
            M[i][j] = np.zeros(G[i].shape)
            Y[i][j] = np.sign(G[i])

    K_max = 10 ** 2
    Convergence_rec = np.zeros(maxiter)

    iter = 0
    while iter < maxiter:

        # update G
        for n in range(N):
            Msum = Msum_fun(M)
            Ysum = Msum_fun(Y)

            Q = tenmat_sb(Z_neq(G, n), 2)
            Q = Q.T

            G[n] = Gfold(np.dot((np.dot((Lambda * tenmat_sb(X, n + 1)), Q.T) + K * Gunfold(
                Msum[n], 1) + Gunfold(Ysum[n], 1)), np.linalg.pinv(
                (Lambda * (np.dot(Q, Q.T)) + 3 * K * np.eye(Q.shape[0], Q.shape[0])))), G[n].shape, 1)

            # update  M

            for j in range(3):
                Df = Gunfold(G[n] - Y[n][j] / K, j)
                M[n][j] = Gfold(Pro2TraceNorm(Df, 1 / K)[0], G[n].shape, j)

        # update X
        lastX = X
        X_hat = coreten2tr(G)
        X = X_hat
        X[W == 1] = T[W == 1]

        # update Y
        for n in range(N):
            for j in range(0, 3):
                Y[n, j] = Y[n, j] + K * (M[n, j] - G[n])
        K = min(K * ro, K_max)

        # evaluation
        G_out = G

        err_x = np.abs(
            (np.linalg.norm(lastX.T.flatten()) - np.linalg.norm(X.T.flatten())) / np.linalg.norm(X.T.flatten()))

        if err_x < tol:
#            print('iteration stop at %f\n' % iter)
            break
        if iter % 100 == 0 or iter == 0:
            Ssum_G = 0  # singular value
            for j in range(N):
                _, vz1, __ = np.linalg.svd(mytenmat(G[0], 1, 1))
                _, vz2, __ = np.linalg.svd(mytenmat(G[0], 2, 1))
                _, vz3, __ = np.linalg.svd(mytenmat(G[0], 3, 1))

                Ssum_G = Ssum_G + np.sum(vz1.T.flatten()) + np.sum(vz2.T.flatten()) + np.sum(vz3.T.flatten())
            f_left = Ssum_G
            f_right = Lambda * (np.linalg.norm(X.T.flatten() - X_hat.T.flatten())) ** 2
            Convergence_rec[iter] = f_left + f_right
#            print('TRLRF: Iter %f, Diff %d, Reg %d, Fitting %d' % (iter, err_x, f_left, f_right))
        iter = iter + 1

    return X, G_out, Convergence_rec
