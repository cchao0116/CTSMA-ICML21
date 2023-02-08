# CTSMA-ICML21
Code for ICML21 paper "Learning Self-Modulating Attention in Continuous Time Space with Applications to Sequential Recommendation"

## Installation

The program requires Python 3.7+ with NumPy, Pandas and Tensorflow 1.x.

## Data Format

The implementation is desiged for top-N recommendations on implicit data, 
and thus it takes Tensorflow-Records as input:
```
seqs_i:   int64, the sequence of item ids
seqs_t: float32, the sequence of purchase timestamps
label:    int64, the output item ids
```

In addition to train/validation/test files, 
the mapping from item-id to mark-id should be also specified 
and stored as scipy-sparse matrix.


## Train and Test

Once the data is ready, it is quite simple to train and evaluate our S2PNM model by running
```
bash runme.sh
```


## Citation

If you find our code useful for your research, please consider cite.

```
@inproceedings{chen2021learning,
  title={Learning Self-Modulating Attention in Continuous Time Space with Applications to Sequential Recommendation},
  author={Chen, Chao and Geng, Haoyu and Yang, Nianzu and Yan, Junchi and Xue, Daiyue and Yu, Jianping and Yang, Xiaokang},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML '21)},
  pages={1606--1616},
  year={2021},
  organization={PMLR}
}
```
