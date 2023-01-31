# Oblique-Forest-AutoEncoders

###### This repository contains the codes for the article *OF-AE: Oblique Forest AutoEncoders* which can be found at *https://arxiv.org/abs/2301.00880*, where we have implemented an unsupervised forest-type autoencoder. This is an extension of the eForest encoder (implemented in *https://github.com/kingfengji/eForest*), consisting of oblique trees (HHCART and RandCART). The implementation is based on finding a sparse solution of a set of linear inequalities consisting of feature values constraints using the optimization package CVXPY. 
###### As noted in the paper, the implementation of the oblique trees is based upon *https://github.com/valevalerio/Ensemble_Of_Oblique_Decision_Trees* and *https://github.com/hengzhe-zhang/scikit-obliquetree*, respectively. 
###### For the setup, one must specify the train and test paths for the custom datasets (for usage, see the file *misc/datasets.py*). Furthermore, inside the codes one must put the absolute path in the command *sys.path.insert* in order to correctly load up the modules. 
