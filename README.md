# Triplet Loss Convolutional Net for Speaker Identification

### Summary

This repository includes my implementation and training/test of a Triplet loss convolutional neural network for the task speaker identification. I used the TIMIT data set (https://catalog.ldc.upenn.edu/LDC93s1) which is unfortunately proprietary and cannot be shared. Therefore, this repository is code only.

The modeling pipeline is as follows
1. Load speeches data
2. Randomly sample multiple one-second segments from each speech file, each has its label being the speaker ID.
3. Transform the segments into melspectrograms
4. Use melspectrograms as image, train a CNN using Triplet loss objective
5. Use the final output embeddings of CNN in a KNN model to make prediction

An illustration of the modeling pipeline
![image.png](attachment:image.png)

### Libraries
- Python 2.7
- NumPy
- SciPy
- Matplotlib
- Theano
- Librosa

### Files
- <b>layers.py</b>: defining fully-connected layers and convolutional+pooling layer
- <b>utility.py</b>: defining utility functions including load data, sample speeches, get precision-recall curve, etc
- <b>melspectrogram demonstration.ipynb</b>: demonstrate building melspectrograms using librosa, numpy, and theano
- <b>speaker identification timit</b>: implementation and testing of triplet loss CNN on TIMIT data
