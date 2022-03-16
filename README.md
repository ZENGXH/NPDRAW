# NP-DRAW: A Non-Parametric Structured Latent Variable Model for Image Generation

This repo contains the official implementation for the [NP-DRAW paper](https://arxiv.org/abs/2106.13435). 

by [Xiaohui Zeng](https://www.cs.utoronto.ca/xiaohui), [Raquel Urtasun](http://www.cs.toronto.edu/~urtasun/), [Richard Zemel](http://www.cs.toronto.edu/~zemel/inquiry/home.php), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), and [Renjie Liao](http://www.cs.toronto.edu/~rjliao/)

## Abstract 
In this paper, we present a non-parametric structured latent variable model for image generation, called **NP-DRAW**, which sequentially draws on a latent canvas in a part-by-part fashion and then decodes the image from the canvas. Our key contributions are as follows. 
1) We propose a non-parametric prior distribution over the appearance of image parts so that the latent variable “what-to-draw” per step becomes a categorical random variable. This improves the expressiveness and greatly eases the learning compared to Gaussians used in the literature. 
2) We model the sequential dependency structure of parts via a Transformer, which is more powerful and easier to train compared to RNNs used in the literature. 
3) We propose an effective heuristic parsing algorithm to pre-train the prior. Experiments on MNIST, Omniglot, CIFAR-10, and CelebA show that our method significantly outperforms previous structured image models like DRAW and AIR and is competitive to other generic generative models. 

Moreover, we show that our model’s inherent compositionality and interpretability bring significant benefits in the low-data learning regime and latent space editing.


## Generation Process 
![prior](https://github.com/ZENGXH/NPDRAW/blob/main/docs/npdraw_prior.gif?raw=true) 

Our prior generate "whether", "where" and "what" to draw per step. If the "whether-to-draw" is true, a patch from the part bank is selected and pasted to the canvas. The final canvas is refined by our decoder. 

## More visualization of the canvas and images 
![twitter-1page](https://user-images.githubusercontent.com/12856437/123703370-473bd500-d832-11eb-8a14-8571bc7caf4a.gif)

## Latent Space Editting 
We demonstrate the advantage of our interpretable latent space via interactively editing/composing the latent canvas. 

![edit](https://user-images.githubusercontent.com/12856437/122693542-47235000-d208-11eb-8d5a-5a26f1edaf33.png)

* Given images A and B, we encode them to obtain the latent canvases. Then we compose a new canvas by placing certain semantically meaningful parts (e.g., eyeglasses, hair, beard, face) from canvas B on top of canvas A. Finally, we decode an image using the composed canvas.

----------------------------
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
```bash
# for mnist
bash scripts/local_sample.sh exp/stoch_mnist/cat_vloc_at/0208/p5s5n36vitBinkl1r1E3_K50w5sc0_gs_difflr_b500/E00550.pth # FID 2.55

# for omniglot
bash scripts/local_sample.sh exp/omni/cat_vloc_at/0208/p5s5n36vitBinkl1r1E3_K50w5sc0_gs_difflr_b500/ckpt_epo799.pth # FID 5.53

# for cifar
bash scripts/local_sample.sh exp/cifarcm/cat_vloc_at/0208/p4s4n64_vitcnnLkl11E3_K200w4sc0_gs_difflr_b150/ckpt_epo499.pth #

# for celeba
bash scripts/local_sample.sh exp/celebac32/cat_vloc_at/0208/p4s4n64_vitcnnLkl0e531E3_K200w4sc0_gs_difflr_b150/ckpt_epo199.pth # FID 41.29
```

## Training 
Use `./scripts/train_$DATASET.sh` to train the model. 

----------------------
* The code in `tool/pytorch-fid/` is adapated from [here](https://github.com/mseitzer/pytorch-fid)
* The transformer code is adapted from [here](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py)

