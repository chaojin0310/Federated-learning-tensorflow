# A Benchmark for Federated Learning implemented on top of LEAF by TensorFlow

## Resources

  * **Homepage:** [leaf.cmu.edu](https://leaf.cmu.edu)
  * **Paper:** ["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)

## Datasets Added

CIFAR-100

  * **Overview:** Image Dataset
  * **Details:** 100 different classes 20 different super classes, images are 3072 pixels(32 * 32 * 3 and preprocess to make them all 224 by 224 pixels), 600 users
  * **Task:** Image Classification

## Notes

- Install the libraries listed in ```requirements.txt```
    - I.e. with pip: run ```pip3 install -r requirements.txt```
- Go to directory of respective dataset for instructions on generating data
    - in MacOS check if ```wget``` is installed and working
- ```models``` directory contains instructions on running baseline reference implementations

## Instructions

For example, to preprocess data such as the `cifar-100` dataset, you could first `cd ./data/cifar-100/ `, then run  `nohup bash ./preprocess.sh -s iid --iu 1.0 --sf 1.0 -k 0 -t sample --tf 0.8 > preprocess_cifar-100_iid.log &` and `nohup bash ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample > preprocess_cifar-100_niid.log &`  for `iid` and `niid` cases, respectively.

To run it, please see `readme.md` in `./models`.

