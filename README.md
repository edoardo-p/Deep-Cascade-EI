# Equivariant Imaging applied to a Deep Cascade of CNNs

This project utilizes equivariant imaging as described by [this paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.pdf) applied to a deep cascade of convolutional neural networks, the model for which has been adapted from [here](https://ieeexplore.ieee.org/document/8067520)

## Requirements

- [PyTorch](https://pytorch.org/) (1.6)

All used packages are listed in the Anaconda environment.yml file. You can create an environment and run

```
conda env create -f environment.yml
```

## Test

We provide the trained models used in the paper which can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1Io0quD-RvoVNkCmE36aQYpoouEAEP5pF?usp=sharing).
Please put the downloaded folder 'ckp' in the root path. Then evaluate the trained models by running

```
python demo_test_ct.py
```

## Train

To train EI, run

```
python demo_train.py
```

or run a bash script to train the models for both CT and inpainting tasks.

```
bash train_paper_bash.sh
```

### Train your models

To train your EI models on your dataset for a specific inverse problem (e.g. inpainting), run

```
python3 demo_train.py --h
```

- Note: you may have to implement the forward model (physics) if you manage to solve a new inverse problem.
- Note: you only need to specify some basic settings (e.g. the path of your training set).
