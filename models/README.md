# SmileyPad: FecNet Training
## Introduction
This part of the repository contains code pertaining to the training of the model at the core of SmileyPad. It is not necessary in order to make SmileyPad work, but is provided mostly for curious developers interested in reproducing the construction of the model.

The model is based on an architecture presented in [[1]](#1). The dataset was also first introduced in the same paper. 

## Using Code
In order to use the code you will need to complete the following steps:

### Initialize facenet-pytorch 
We are using facenet-pytorch as the backbone of our neural network. We need to initialize this submodule:

``
git submodule init

git submodule update
``

### Download FEC Dataset
This is the dataset presented in the mentioned paper. To download it, you will need to agree to the authors' terms.
<b>To download the dataset</b>: [FEC Dataset](https://ai.google/tools/datasets/google-facial-expression/)
Once you download the dataset, extract it into a subdirectory named "FEC_dataset" in this directory (meaning there will be a directory called "FEC_dataset" which will contain the dataset in a subdirectory also named "FEC_dataset").

### Preprocess Dataset
run

``
python preprocess_dataset.py
``

This one is tricky. Right now the preprocess seems to hang after ~10000-20000 iterations, so you can use the `--start` flag to pick up from where you stopped (e.g. `python preprocess_dataset.py --start 10000` to continue processing the dataset from iteration 10000). It is annoying and should be fixed but for now, this is what we got. Notice it is not a problem to re-do some iterations. 

Notice: 
* This script is randomized, but you provide a seed (0, by default), so as long as you keep the same seed, `--start` will work as expected, otherwise you cannot expect it to behave in a proper way. The main takeaway is: if you don't touch the seed, you don't need to worry about undefined behavior

* The code hangs when you send keyboard interrupts, so you will have to kill it in a more aggressive way. Again, this is to be fixed in the future.

### Train

Once completed of the preprocess step, you are ready to go with the training step.
Simply run:

``
python train.py
``



## References
<a id="1"> [1] </a> [A Compact Embedding for Facial Expression Similarity](http://arxiv.org/abs/1811.11283). Raviteja Vemulapalli and Aseem Agarwala. 2018