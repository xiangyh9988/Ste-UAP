# About the project
This is a project to repoduce "[Steganographic universal adversarial perturbations](https://www.sciencedirect.com/science/article/pii/S016786552030146X)" (**unofficially**).

Apologize for my terrible codes in advance. The steganography transform is in `transform.py` and the test is in `attack.py`. For more details, please refer to `data.py` and `models.py`,

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
> Note: I doubt that there are some typos or mistakes in the original paper, such as the name of DWT components and the HL component shown in Fig.2. I will check that further and then refine codes.

## 2. Adversarial Attack
~~Not implemented yet.~~

The original paper test on ImageNet validation set. Besides, I intend to test on CIFAR-10 and MNIST in the future.

### 1. Attack typical standard classification models.

ResNet50, VGG16, Inception-V3 and MobileNet-V2 are attacked as the original paper does. Besides, I test on DenseNet121. These models are loaded from `torchvision.models`.

|     | ResNet50  |  VGG16  |  Inception-V3  |  MobileNet-V2  |  DenseNet121  |
|  ----  | ----  | ---- | ---- | ---- | ---- |  
| **Paper** | --/71.76% | --/73.88% | --/64.59% | --/77.52% | --/-- |
| clean | 77.20%/-- | 72.70%/-- | 77.30%/-- | 70.70%/-- | 74.00%/-- |
| DWT-IDWT | 76.60%/2.10% | 72.70%/1.30%| 76.30%/5.80% | 70.80%/2.20% | 71.40%/0.70% |
| HH | 63.70%/30.50% | 57.80%/35.20% | 76.30%/8.80% | 60.20%/31.90% | 71.10%/18.00% |
| **DWT-SVD** | 56.10%/39.90% | 41.60%/54.70% | 73.20%/13.50% | 45.40%/49.70% | 63.50%/29.00% |
| All high | 18.90%/80.60% | 1.90%/98.20% | 59.80%/34.60% | 5.90%/93.20% | 26.70%/70.50% |

Each cell is *accuracy/fooling rate* and '--' means no data here.

Note that **Paper** represents the results from the original paper and **DWT-SVD** are my reproduction. As you can see, there is a large margin between the results. The only difference is that I use so-called HL in paper as LH in `pywt` and vice versa, because I see another attack in transform domain (i.e., WaveTransform) doing like this. And the swap of LH and HL makes my metrics even better. So, I'm confused about the failure. I will appreciate it if someone can help me.

An intriguing point is that the inverse DWT via LL of host and other high frequency components of secret can achieve the best attack performance, i.e., highest fooling rate. However, both *DWT-SVD* and *All high* make the $l_2$ and $l_\infty$ norm very large, and the distortion of *All high* is significantly perceptible. Besides, DWT-IDWT, namely reconstruction via host's components, causes some information loss as the change of accuracy and fooling rate show.


**TODO:**

2. Attack some defense models (not yet determined what models).