# AMLSII_22-23_SN22081179
ELEC 0135 Applied Machine Learning Systems II aimed at solving the NTIRE 2017 super resolution challenge


## 1. Prerequisites
To Begin with, it is required to clone this project or download to your computer or server. The structure of the folders are as follows:

### AMLSII_22-23_SN22081179

* [A/](./A)
  * [datasets.py](./A/datasets.py)
* [B/](./B)
  * [datasets.py](./B/datasets.py)
* Datasets/
  * DIV2K_train_HR/
  * DIV2K_train_LR_bicubic_X2/
  * DIV2K_train_LR_unknown/
  * DIV2K_valid_HR/
  * DIV2K_valid_LR_bicubic_X2/
  * DIV2K_valid_LR_unknown/
* [src/](./src)
  * [models.py](./src/models.py)
  * [utils.py](./src/utils.py)
* [checkpoint_srgan_A.pth.tar](./checkpoint_srgan_A.pth.tar)
* [checkpoint_srgan_B.pth.tar](./checkpoint_srgan_B.pth.tar)
* [checkpoint_srresnet_A.pth.tar](./checkpoint_srresnet_A.pth.tar)
* [checkpoint_srresnet_B.pth.tar](./checkpoint_srresnet_B.pth.tar)
* [eval_srgan_A.py](./eval_srgan_A.py)
* [eval_srgan_B.py](./eval_srgan_B.py)
* [eval_srresnet_A.py](./eval_srresnet_A.py)
* [eval_srresnet_B.py](./eval_srresnet_B.py)
* [super_resolve_A.py](./super_resolve_A.py)
* [train_srgan_A.py](./train_srgan_A.py)
* [train_srgan_B.py](./train_srgan_B.py)
* [train_srresnet_A.py](./train_srresnet_A.py)
* [train_srresnet_B.py](./train_srresnet_B.py)

The folder Datasets is not included, you can either download the required Datesets according to the file tree or you can download the zip file [here](https://drive.google.com/file/d/10lEX7Jo9BJv3Ve2bW1-RuMCyoY3MJ0l8/view?usp=sharing). The checkpoints can also be downloaded [here](https://drive.google.com/file/d/1IwYsMsFN71HugFLTRjOX1CGaXkLCsBis/view?usp=sharing) in case you accidentally updated the original one during the training process.

### The environment

My advice is to create a new conda environment from the `environment.yml` file in this repo [environment.yml](./environment.yml)
You can simply do it by: 

```bash
conda env create -f environment.yml
```

## 2. How to check the result of this project

### If your server or computer is GPU ready

(For the project I used turin.ee.ucl.ac.uk, I checked whether it woeks on this server, I strongly suggest that you test the project on the server)
You can simply check by input:

```python
torch.cuda.is_available()
```

If the output is `True`, then congratulation that you can start from the training by exectuting the corresponding file to solve the Task A and B. However, you should always train the SRResnet before training the SRGAN as the weights of SRResnet are needed when training SRGAN.

#### This means for task A the training sequence should be:

[train_srresnet_A.py](./train_srresnet_A.py) -> [train_srgan_A.py](./train_srgan_A.py)

#### For task B the training sequence should be:

[train_srresnet_B.py](./train_srresnet_B.py) -> [train_srgan_B.py](./train_srgan_B.py)

### You can also just check the result by executing the eval funciton

#### For task A it would be executing:


* [eval_srresnet_A.py](./eval_srresnet_A.py)
* [eval_srgan_A.py](./eval_srgan_A.py)
`

#### For task B it would be executing:


* [eval_srresnet_B.py](./eval_srresnet_B.py)
* [eval_srgan_B.py](./eval_srgan_B.py)

