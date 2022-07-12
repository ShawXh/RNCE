This repo includes the implementation of methods with RNCE and an re-implementation of RNS.

# Files
We adopt the proposed regualrizer on several methods including LINE, node2vec, VERSE. The files are:
- base-rnce.cpp (for vanilla-x, vanilla-h, LINE-1st, LINE-2nd, LINE)
- node2vec-rnce.cpp (for node2vec)
- verse-rnce-sgd.cpp (for verse with SGD)
- verse-rnce-mbsgd.cpp (for verse with Mini-batch SGD)
- rns.cpp (for RNS)

# Compile
```
sh compile.sh
```

# RNCE
The implementation of TKDE'22 paper [''Learning Regularized Noise ContrastiveEstimation for Robust Network Embedding''](https://shawxh.github.io/assets/gomune.pdf).

## Usage Example
```
./$method -input network_path -reg $distanc_func -beta 0.01 -emb-u $emb_x_path -emb-v $emb_h_path -samples 100 -negatives 5 
```

For full usage, please refer to the `main()` function in cpp code.

# RNS
We re-implement the paper AAAI19 paper ''[Robust Negative Sampling for Negative Embedding](https://ojs.aaai.org/index.php/AAAI/article/view/4187/4065)'' (including the embedding penalty and the adaptive negative sampler).

## Usage example
```
./rns -input $network_path -rns 3 -emb-u $emb_x_path -emb-v $emb_h_path -samples 100 -negatives 5
```

Notes:
- rns=0 for without rns
- rns=1 for only embedding norm penalty
- rns=2 for only adapative negative sampler
- rns=3 for both embedding norm penalty and adapative negative sample

# Experiments

## Node Classification
Please refer to [code](https://github.com/ShawXh/Evaluate-Embedding)

## Network Reconstruction
First,
```
cd eval_network_reconstruction
```

and then,
```python
python network_reconstruction.py --emb1 $emb_x_path --emb2 $emb_h(x)_path --net $network_path
```

# Citation
```
@ARTICLE{rnce,
  author={Xiong, Hao and Yan, Junchi and Huang, Zengfeng},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Learning Regularized Noise Contrastive Estimation for Robust Network Embedding}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TKDE.2022.3148284}}
```
