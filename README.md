# NP-DRAW: A Non-Parametric Structured Latent Variable Model for Image Generation

This repo contains the official implementation for the paper [NP-DRAW](https://www.cs.utoronto.ca/~xiaohui/paper/npdraw/npdraw.pdf). 
[[paper]](https://www.cs.utoronto.ca/~xiaohui/paper/npdraw/npdraw.pdf) | [[supp]](https://www.cs.utoronto.ca/~xiaohui/paper/npdraw/npdraw_supp.pdf)  

by [Xiaohui Zeng](https://www.cs.utoronto.ca/xiaohui), [Raquel Urtasun](http://www.cs.toronto.edu/~urtasun/), [Richard Zemel](http://www.cs.toronto.edu/~zemel/inquiry/home.php), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), and [Renjie Liao](http://www.cs.toronto.edu/~rjliao/)

## Dependencies
```bash
# the following command will install torch 1.6.0 and other required packages 
conda env create -f environment.yml # edit the last link in the yml file for the directory
conda activate npdraw 
```
## Pretrained Model 
Pretrained model will be available [here](https://drive.google.com/drive/folders/1jTlN6dWv9MnOd7Jo5H5yMpc-pFErYLS1?usp=sharing) 
To use the pretrained models, download the `zip` file under `exp` folder and unzip it. For expample, with the `cifar.zip` file we will get `./exp/cifarcm/cat_vloc_at/` and `./exp/cnn_prior/cifar/`. 

#### Testing the pretrained NPDRAW model:

* before running the evaluation, please also download the stats on the test set from [google-drive](https://drive.google.com/file/d/1U3sBE2kbhFdutTSLj26oHnGIYiXAP6OF/view?usp=sharing), and run 
```
mkdir datasets 
mv images.tar.gz datasets 
cd datasets 
tar xzf images.tar.gz 
``` 

The following commands test the FID score of the NPDRAW model. 
The commands output the CD and EMD on the test/validation sets.
```bash
# Usage:
bash scripts/local_sample.sh exp/stoch_mnist/cat_vloc_at/0208/p5s5n36vitBinkl1r1E3_K50w5sc0_gs_difflr_b500/ckpt_epo799.pth 
bash scripts/local_sample.sh exp/cifarcm/cat_vloc_at/0208/p4s4n64_vitcnnLkl11E3_K200w4sc0_gs_difflr_b150/ckpt_epo499.pth 
```

## Training 
Use `./scripts/train_$DATASET.sh` to train the model. 

----------------------
* The code in `tool/pytorch-fid/` is adapated from [here](https://github.com/mseitzer/pytorch-fid)
* The transformer code is adapted from [here](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py)
