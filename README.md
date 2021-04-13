# GAN_metrics
This repository provides the code for PSNR, SSIM, L1 error, L2 error etc.

Requirement
----
- Python 3
- Numpy
- Tensorflow
- OpenCV-Python

Directory
---
```
path
└── to
    └── image
         └── folder
                └── original
                        ├── 0.jpg
                        └── 1.jpg
                └── ours
                        ├── 0.jpg
                        └── 1.jpg
```

PSNR : Peak Signal-to-Noise Ratio
----
Peak signal-to-noise ratio, often abbreviated PSNR, is an engineering term for the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. <br>
<img src="/img/PSNR.PNG"></img><br/>
<img src="/img/PSNR2.png"></img>

SSIM : Structural similarity
----
The difference with respect to other techniques mentioned previously such as MSE or PSNR is that these approaches estimate absolute errors; on the other hand, SSIM is a perception-based model that considers image degradation as perceived change in structural information, while also incorporating important perceptual phenomena, including both luminance masking and contrast masking terms. Structural information is the idea that the pixels have strong inter-dependencies especially when they are spatially close. These dependencies carry important information about the structure of the objects in the visual scene.<br>
![Alt text](/img/SSIM.PNG)


L1 error
---
![Alt text](/img/L1.PNG)

L2 error
---
![Alt text](/img/L2.PNG)
