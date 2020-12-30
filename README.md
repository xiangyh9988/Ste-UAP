# About the project
This is a project to repoduce "[Steganographic universal adversarial perturbations](https://www.sciencedirect.com/science/article/pii/S016786552030146X)" (**unofficially**).

# Implementation
There are two parts needed to be implemented, i.e., generating the universal adversarial perturbation and attacking different classification models.

To evaluate the robustness of generated adversarial examples, it is necessary to attack some defense models (which is not done in the original paper).

## 1. Ste-UAP Generation
![Pipeline](https://github.com/xiangyh9988/Image-Hosting/blob/main/imgs/image-20201227153444844.png?raw=true)

Adversarial examples (i.e., stego images) are generated via discrete wavelet transform (**DWT**) with the 'Haar' filter and singular value decomposition (**SVD**).

Let $x_1$ be the host image (benign example) and $x_2$ be the secret image (the source of perturbation).

The procedure is as follow and please refer to *Sec 4.2-4.4* for details. 
> Fucking formulas.

1. DWT on $x_1$ and $x_2$ separately, each of which is decomposed into 4 components, denoted by LL, LH, HL, HH.
2. SVD on $x_{1LL}$ and $x_{2LL}$ to get $S_{X_{1LL}}$ and $S_{X_{2LL}}$ and fuse them.
3. Use LH of $x_1$ as LH of the stego image.
4. Rotate HL at 4 different angles and average them. Then fuse the averaged component and HL of $x_1$ as HL of the stego image.
5. Use HH of $x_2$ as HH of the stego image.
6. IDWT on 4 components obtained above to reconstruct the stego image.

Then you can get an adversarial example.
> Note: I doubt that there are some typos or mistakes in the original paper, such as the name of DWT components and the HL component showed in Fig.2. I will check that further and then refine codes.

## 2. Adversarial Attack
Not implemented yet. 

The original paper test on ImageNet validation set. Besides, I intend to test on CIFAR-10 and MNIST. 

**TODO:**

1. Attack typical standard classification models, e.g. VGG16, ResNet50.
2. Attack some defense models (not yet determined what models).