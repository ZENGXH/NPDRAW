# NP-DRAW: A Non-Parametric Structured Latent Variable Model for Image Generation 
by [Xiaohui Zeng](https://www.cs.utoronto.ca/xiaohui), [Raquel Urtasun](http://www.cs.toronto.edu/~urtasun/), [Richard Zemel](http://www.cs.toronto.edu/~zemel/inquiry/home.php), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), and [Renjie Liao](http://www.cs.toronto.edu/~rjliao/)

[[paper]](https://www.cs.utoronto.ca/~xiaohui/paper/npdraw/npdraw.pdf) | [[supp]](https://www.cs.utoronto.ca/~xiaohui/paper/npdraw/npdraw_supp.pdf)

## Abstract 
In this paper, we present a non-parametric structured latent variable model for image generation, called **NP-DRAW**, which sequentially draws on a latent canvas in a part-by-part fashion and then decodes the image from the canvas. Our key contributions are as follows. 
1) We propose a non-parametric prior distribution over the appearance of image parts so that the latent variable “what-to-draw” per step becomes a categorical random variable. This improves the expressiveness and greatly eases the learning compared to Gaussians used in the literature. 
2) We model the sequential dependency structure of parts via a Transformer, which is more powerful and easier to train compared to RNNs used in the literature. 
3) We propose an effective heuristic parsing algorithm to pre-train the prior. Experiments on MNIST, Omniglot, CIFAR-10, and CelebA show that our method significantly outperforms previous structured image models like DRAW and AIR and is competitive to other generic generative models. 

Moreover, we show that our model’s inherent compositionality and interpretability bring significant benefits in the low-data learning regime and latent space editing.


## Generation Process 
![prior](https://github.com/ZENGXH/NPDRAW/blob/main/docs/npdraw_prior.gif) 

Our prior generate "whether", "where" and "what" to draw per step. If the "whether-to-draw" is true, a patch from the part bank is selected and pasted to the canvas. The final canvas is refined by our decoder. 

## More visualization of the canvas and images 
![mnist](https://user-images.githubusercontent.com/12856437/122693351-7be2d780-d207-11eb-96e9-1ee965bdd3b8.gif)
![omni](https://user-images.githubusercontent.com/12856437/122693350-78e7e700-d207-11eb-9d2b-035d03a17d9f.gif)
![cifar](https://user-images.githubusercontent.com/12856437/122693478-062b3b80-d208-11eb-9bdd-4805adc1883c.png)
![celeba](https://user-images.githubusercontent.com/12856437/122693481-09262c00-d208-11eb-9ca1-726c7448452f.png)

## Latent Space Editting 
We demonstrate the advantage of our interpretable latent space via interactively editing/composing the latent canvas. 

![edit](https://user-images.githubusercontent.com/12856437/122693542-47235000-d208-11eb-8d5a-5a26f1edaf33.png)

* Given images A and B, we encode them to obtain the latent canvases. Then we compose a new canvas by placing a portion of canvas B on top of canvas A. Finally, we decode an image using the composed canvas.
* we can synthesize new images by placing certain semantically meaningful parts (e.g., eyeglasses, hair, beard, face) from B to A.
