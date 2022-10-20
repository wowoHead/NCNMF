import numpy as np
import math
import random
from sys import float_info


def CCNMF(A, S_la, S, max_iter, k, l, alpha, cita, beta):
    # A--Adjacency matrix
    # S--Similarity matrix
    # k--Number of communities
    # l--lambda
    eps = float_info.epsilon
    n = A.shape[0]
    D = np.diagflat(np.sum(A, axis=1))
    A2 = A + alpha * np.eye(n)
    N = (1 - beta) * np.ones((n, k))
    B = np.ones((k,1))
    C = np.ones((n,1))
    E = np.ones((k,n))
    H = np.random.rand(n, k)
    L = S_la - S
    # obj = np.linalg.norm(A-H.dot(H.T), ord='fro')**2 + l*((H.T).dot(L.dot(H))).trace() + alpha*(H.dot(E-H.T).dot(D)).trace()
    for niter in range(1, max_iter):
        # updata U
        # print('iteration=',niter,'obj=',obj)
        # last_obj = obj
        # cita = cita * 1.02
        up = 4.0*A.dot(H) + 2*l*S.dot(H) + 2*alpha*D.dot(H) + 2*cita*C.dot(B.T)
        down = 4*H.dot((H.T).dot(H)) + 2*l*S_la.dot(H) + alpha*D.dot(E.T) + 2*cita*H.dot(B.dot(B.T))
        H = H * (N + beta * up / np.maximum(down, eps))
        # obj = np.linalg.norm(A-H.dot(H.T), ord='fro')**2 + l*((H.T).dot(L.dot(H))).trace() + alpha*(H.dot(E-H.T).dot(D)).trace()
    # print(cita)
    return H