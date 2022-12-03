import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import NewJaccard
# import sys
# sys.path.append('/home/ssx/code/data_processing')
# import texas

def get_3_order_similarity(A):
    G = nx.Graph(A)
    n = len(A)
    d_mean = len(G.edges) / n
    neighbor_1_set = []
    neighbor_2_set = []
    neighbor_3_set = []

    for i in range(n):
        neighbor_1_set.append(get_1_neighbor(A,i))

    neighbor_2_set = get_2_neighbor(A)

    neighbor_3_set = get_3_neighbor(A)

    S = np.zeros((n,n))

    for i in range(n):
        i_1_neighbor = neighbor_1_set[i]
        i_2_neighbor = neighbor_2_set[i]
        i_3_neighbor = neighbor_3_set[i]
        for j in range(n):
            if i==j:
                continue
            if nx.has_path(G,i,j) == False:
                continue
            j_1_neighbor = neighbor_1_set[j]
            j_2_neighbor = neighbor_2_set[j]
            j_3_neighbor = neighbor_3_set[j]
            S[i][j] = len(set(i_1_neighbor) & set(j_1_neighbor)) / len(set(i_1_neighbor) | set(j_1_neighbor)) * (1/d_mean)
            S[i][j] = S[i][j] + len(set(i_1_neighbor) & set(j_2_neighbor)) / len(set(i_1_neighbor) | set(j_2_neighbor)) * (1/(d_mean*d_mean))
            S[i][j] = S[i][j] + len(set(i_2_neighbor) & set(j_1_neighbor)) / len(set(i_2_neighbor) | set(j_1_neighbor)) * (1/(d_mean*d_mean))
            if len(set(i_2_neighbor) | set(j_2_neighbor)) != 0:
                S[i][j] = S[i][j] + len(set(i_2_neighbor) & set(j_2_neighbor)) / len(set(i_2_neighbor) | set(j_2_neighbor)) * (1/(d_mean*d_mean*d_mean*d_mean))
            S[i][j] = S[i][j] + len(set(i_1_neighbor) & set(j_3_neighbor)) / len(set(i_1_neighbor) | set(j_3_neighbor)) * (1/(d_mean*d_mean*d_mean))
            if len(set(i_2_neighbor) | set(j_3_neighbor)) != 0:
                S[i][j] = S[i][j] + len(set(i_2_neighbor) & set(j_3_neighbor)) / len(set(i_2_neighbor) | set(j_3_neighbor)) * (1/d_mean**6)
            S[i][j] = S[i][j] + len(set(i_3_neighbor) & set(j_1_neighbor)) / len(set(i_3_neighbor) | set(j_1_neighbor)) * (1/d_mean**3)
            if len(set(i_3_neighbor) | set(j_2_neighbor)) != 0:
                S[i][j] = S[i][j] + len(set(i_3_neighbor) & set(j_2_neighbor)) / len(set(i_3_neighbor) | set(j_2_neighbor)) * (1/d_mean**6)
            if len(set(i_3_neighbor) | set(j_3_neighbor)) != 0:
                S[i][j] = S[i][j] + len(set(i_3_neighbor) & set(j_3_neighbor)) / len(set(i_3_neighbor) | set(j_3_neighbor)) * (1/d_mean**9)
    return S


def get_2_order_similarity(A):
    G = nx.Graph(A)
    n = len(A)
    d_mean = len(G.edges) / n
    neighbor_1_set = []
    neighbor_2_set = []
    for i in range(n):
        neighbor_1_set.append(get_1_neighbor(A,i))

    neighbor_2_set = get_2_neighbor(A)

    S = np.zeros((n,n))

    for i in range(n):
        i_1_neighbor = neighbor_1_set[i]
        i_2_neighbor = neighbor_2_set[i]
        for j in range(n):
            if i == j:
                continue
            j_1_neighbor = neighbor_1_set[j]
            j_2_neighbor = neighbor_2_set[j]
            S[i][j] = len(set(i_1_neighbor) & set(j_1_neighbor)) / len(set(i_1_neighbor) | set(j_1_neighbor)) * (1/d_mean)
            S[i][j] = S[i][j] + len(set(i_1_neighbor) & set(j_2_neighbor)) / len(set(i_1_neighbor) | set(j_2_neighbor)) * (1/(d_mean*d_mean))
            S[i][j] = S[i][j] + len(set(i_2_neighbor) & set(j_1_neighbor)) / len(set(i_2_neighbor) | set(j_1_neighbor)) * (1/(d_mean*d_mean))
            if len(set(i_2_neighbor) | set(j_2_neighbor)) != 0:
                S[i][j] = S[i][j] + len(set(i_2_neighbor) & set(j_2_neighbor)) / len(set(i_2_neighbor) | set(j_2_neighbor)) * (1/(d_mean*d_mean*d_mean*d_mean))

    return S

def get_3_neighbor(A):
    n = len(A)
    neighbor_1_set = []
    neighbor_2_set = []
    neighbor_3_set = []

    for i in range(n):
        neighbor_1_set.append(get_1_neighbor(A,i))

    neighbor_2_set = get_2_neighbor(A)

    for i in range(n):
        temp = []
        for j in neighbor_2_set[i]:
            j_neighbor = get_1_neighbor(A,j)
            for u in j_neighbor:
                temp.append(u)
        temp = set(temp)
        neighbor_3_set.append(list(temp - set(neighbor_1_set[i]) - set(neighbor_2_set[i])))
    return neighbor_3_set

def get_2_neighbor(A):
    n = len(A)
    neighbor_1_set = []
    neighbor_2_set = []
    for i in range(n):
        neighbor_1_set.append(get_1_neighbor(A,i))
    
    for i in range(n):
        temp = []
        for j in neighbor_1_set[i]:
            j_neighbor = get_1_neighbor(A,j)
            for u in j_neighbor:
                temp.append(u)
        temp = set(temp)
        neighbor_2_set.append(list(temp - set(neighbor_1_set[i])))
    return neighbor_2_set


def get_1_neighbor(A,v):
    n = len(A)
    neighbor = []
    neighbor.append(v)
    for i in range(n):
        if A[v][i] == 1:
            neighbor.append(i)
    return neighbor


