# GAN Metric
import numpy as np
import tensorflow as tf
import math
import cv2
import glob
import os

# PSNR 
# bigger is better, good quality
def psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0: # mse == 0 means lossless
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# SSIM
# bigger is better, good quality
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def l1_loss(img1, img2): # smaller is better, good quality
    img1 = img1.astype(np.float64)
    img1 = img1/255.0
    img2 = img2.astype(np.float64)
    img2 = img2/255.0
    l1 = np.mean(np.abs(img1-img2))
    return l1

def l2_loss(img1, img2): # smaller is better, good quality
    img1 = img1.astype(np.float64)
    img1 = img1/255.0
    img2 = img2.astype(np.float64)
    img2 = img2/255.0

    l2 = np.mean(np.square(img1-img2))
    return l2

def Average(lst):
    return sum(lst) / len(lst)

def main():
    i = 0
    psnr_list = []
    ssim_list = []
    L1_list = []
    L2_list = []

    original_path = './training_data/validation/original/*.jpg'
    ours_path = './training_data/validation/ours/*.jpg'

    original_list = glob.glob(original_path)
    ours_list = glob.glob(ours_path)

    for i in range(1000):
        original = cv2.imread(original_list[i])
        # original_crop = original[180:180+170, 250:250+170] #cropped
        
        ours = cv2.imread(ours_list[i])
        # ours_crop = ours[180:180+170, 250:250+170] #cropped
        
        print(original_list[i], ours_list[i])
      
        value1 = psnr(original, ours)
        psnr_list.append(value1)
        value2 = calculate_ssim(original, ours)
        ssim_list.append(value2)
        value3 = l1_loss(original, ours)
        L1_list.append(value3)
        value4 = l2_loss(original, ours)
        L2_list.append(value4)

        i += 1

    average1 = Average(psnr_list)
    average2 = Average(ssim_list)
    average3 = Average(L1_list)
    average4 = Average(L2_list)   

    print("pre Average of PSNR =", round(average1, 2))
    print("pre Average of SSIM =", round(average2, 4))
    print("pre Average of L1 loss =", round(average3, 4))
    print("pre Average of L2 loss =", round(average4, 4))


if __name__ == "__main__":
    main()

