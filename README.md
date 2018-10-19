## MLDG
This is a code sample for the paper "Learning to Generalize: Meta-Learning for Domain Generalization" https://arxiv.org/pdf/1710.03463.pdf


This code is the MLP version of MLDG with one-hidden layer, whose inputs are the features extracted for PACS.
The baseline is the one for the sanity check without the meta-train and meta-val losses.



## Requirements
Python 2.7 

Pytorch 0.3.1

## Run the baseline
Please download the data first, the data is the deep features extracted from ImageNet pretrained ResNet18, then

sh run_baseline.sh 'data_root/'         # data_root is the folder path where you download your data to.

## Run the MLDG

sh run_mldg.sh 'data_root/'

## Bibtex
```
 @inproceedings{Li2018MLDG,
   title={Learning to Generalize: Meta-Learning for Domain Generalization},
   author={Li, Da and Yang, Yongxin and Song, Yi-Zhe and Hospedales, Timothy},
  	booktitle={AAAI Conference on Artificial Intelligence},
  	year={2018}
 }
 ```
 
 ## Your own data
 Please tune the 'meta_step_size' and 'meta_val_beta' for your own data, this parameter is related to 'alpha' and 'beta' in paper which should be tuned for different cases.
