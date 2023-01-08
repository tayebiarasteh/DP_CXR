# Private, fair and still accurate: Building an AI model for chest X-ray under patient privacy guarantees


[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

Overview
------

* This is the official repository of the paper [**Private, fair and still accurate: Building an artificial intelligence model for chest radiographs under patient privacy guarantees**]().
* Pre-print version: []()

------

Introduction
------
Differentially-private training of advanced artificial intelligence models for diagnosis of chest radiographs with privacy-preserving techniques yields results with negligible performance and fairness trade-offs on large real-world datasets.


![](./intro.png)

------
### Prerequisites

The software is developed in **Python 3.10**. For the deep learning, the **PyTorch 1.13** framework is used. The DP code was developed using **Opacus 1.3.0**.



Main Python modules required for the software can be installed from ./requirements in three stages:

1. Create a Python3 environment by installing the conda `environment.yml` file:

```
$ conda env create -f environment.yml
$ source activate DP_CXR
```


2. Install the remaining dependencies from `requirements.txt`.


**Note:** These might take a few minutes.

------
Code structure
---

Our source code for differential privacy as well as training and evaluation of the deep neural networks, image analysis and preprocessing, and data augmentation are available here.

1. Everything can be run from *./main_2D_DP.py*. 
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./configs/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

2. The rest of the files:
* *./data/* directory contains all the data preprocessing, augmentation, and loading files.
* *./Train_Valid_DP.py* contains the training and validation processes.
* *./Prediction_DP.py* all the prediction and testing processes.

------
### In case you use this repository, please cite the original paper:

