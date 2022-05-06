# Equivariant Imaging applied to a Deep Cascade of CNNs

This project utilizes equivariant imaging as described by [this paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.pdf) applied to a deep cascade of convolutional neural networks, the model for which has been adapted from [here](https://ieeexplore.ieee.org/document/8067520)

## Requirements

All used packages are listed in the Anaconda environment.yml file. You can create an environment and run

```
conda env create -f environment.yml
```

## Test

To test a model, simply run the following command with the appropriate filename
```
python test.py --ckp "./ckp/<filename>"
```
By default, it will load the provided `ckp_final.pth.tar` file.

Additional flags

- `--model-name`: Text displayed on top of the output image
- `--sample-to-show`: Index of test images to display to the screen
- `-h`: Shows help for the above flags

## Train

To train EI, run

```
python train.py
```

Additional flags

- `--schedule`: List of epochs when to drop the learning rate. Default
- `--cos`: Use cosine decay for learning rate and overrides schedule if set
- `--epochs`: Number of training epochs to perform
- `--lr`: Initial learning rate
- `--wd`: Initial weight decay
- `-b`: Batch size
- `--ckp-interval`: How often to save. Regardless, the model will be saved once training is finished
- `--dataset`: Path to the dataset used. Has to be a MATLAB file
- `--ei-trans`: Number of transformations
- `--ei-alpha`: Equivariance strength
- `--views`: Number of subsample views for radon transform
- `-h`: Shows help for the above flags

All of these have default values and the code will still work if you run the command above.
