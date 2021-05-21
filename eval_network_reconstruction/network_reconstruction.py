import argparse
from utils import *

def parse_args():
    '''
    Running for Single-Net embedding.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb1', type=str, default='', help="/data/emb1.txt")
    parser.add_argument('--emb2', type=str, default='', help="/data/emb2.txt")
    parser.add_argument('--net', type=str, default='', help="/data/blog-net.txt")
    return parser.parse_args()

args = parse_args()

print("reading emb")
emb1 = read_emb(args.emb1)
emb2 = read_emb(args.emb2)
print("reading net")
net = read_net(args.net)
print("calculating dist")
dist = euclidean_dist_2(emb1, emb2)

print("calculating precision")
print("Network Reconstruction Precision:", precision(dist, net))
