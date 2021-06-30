import numpy as np
from collections import defaultdict

def read_emb(file_name):
    with open(file_name, "r") as f:
        ln = f.readline()
        n, d = list(map(int, ln.strip().split(" ")))
        emb = np.zeros((n, d))
        for ln in f.readlines():
            ln = ln.strip().split(" ")
            nd = int(ln[0])
            emb[nd] = np.array(list(map(float, ln[1:])))
    return emb

def read_net(file_name):
    net = defaultdict(dict)
    with open(file_name, "r") as f:
        for ln in f.readlines():
            ln = ln.strip().split(" ")
            n1, n2 = list(map(int, ln))[:2]
            net[n1][n2] = 1
            net[n2][n1] = 1
    return net

def euclidean_dist_1(emb1, emb2):
    N = emb1.shape[0]
    dist = np.zeros((N, N))
    for nd in range(N):
        emb = emb1[nd]
        d = emb2 - emb
        d = np.sum(d * d, axis=1)
        dist[nd] = d
    return dist

def euclidean_dist_2(emb1, emb2):
    N1 = np.sum(emb1 * emb1, axis=1)
    NN = emb1.dot(emb2.transpose())
    N2 = np.sum(emb2 * emb2, axis=1)
    return -2 * NN + np.expand_dims(N1, 1) + N2

def precision(dist, net):
    N = dist.shape[0]
    dist = dist.reshape(N * N)
    rank = np.argsort(dist)
    M = 0
    for nd in net:
        M += len(net[nd])
    rank = rank[:M]

    cnt = 0
    for i in range(M):
        a = int(rank[i] / N)
        b = int(rank[i] % N)
        if b in net[a]:
            cnt += 1
    return cnt / M


