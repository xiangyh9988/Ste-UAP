import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from utils.imutils import *


def DWT_SVD(host_img, secret_img):
    # host_img = cv2.resize(host_img, (600, 600))
    # secret_img = cv2.resize(secret_img, (300, 300))
    imshow_opencv('host', host_img)
    imshow_opencv('secret', secret_img)

    # DWT on host x1
    host_b, host_g, host_r = cv2.split(host_img)
    # dwt: LL, HL, LH, HH
    LL_host_b, (HL_host_b, LH_host_b, HH_host_b) = pywt.dwt2(host_b, 'haar')
    LL_host_g, (HL_host_g, LH_host_g, HH_host_g) = pywt.dwt2(host_g, 'haar')
    LL_host_r, (HL_host_r, LH_host_r, HH_host_r) = pywt.dwt2(host_r, 'haar')

    # DWT on secret x2
    secret_b, secret_g, secret_r = cv2.split(secret_img)
    LL_secret_b, (HL_secret_b, LH_secret_b, HH_secret_b) = pywt.dwt2(secret_b, 'haar')
    LL_secret_g, (HL_secret_g, LH_secret_g, HH_secret_g) = pywt.dwt2(secret_g, 'haar')
    LL_secret_r, (HL_secret_r, LH_secret_r, HH_secret_r) = pywt.dwt2(secret_r, 'haar')

    # fusion of LL component
    u_LL_host_b, s_LL_host_b, v_LL_host_b = np.linalg.svd(LL_host_b, full_matrices=1, compute_uv=1)
    u_LL_host_g, s_LL_host_g, v_LL_host_g = np.linalg.svd(LL_host_g, full_matrices=1, compute_uv=1)
    u_LL_host_r, s_LL_host_r, v_LL_host_r = np.linalg.svd(LL_host_r, full_matrices=1, compute_uv=1)
    u_LL_secret_b, s_LL_secret_b, v_LL_secret_b = np.linalg.svd(LL_secret_b, full_matrices=1, compute_uv=1)
    u_LL_secret_g, s_LL_secret_g, v_LL_secret_g = np.linalg.svd(LL_secret_g, full_matrices=1, compute_uv=1)
    u_LL_secret_r, s_LL_secret_r, v_LL_secret_r = np.linalg.svd(LL_secret_r, full_matrices=1, compute_uv=1)

    s_LL_host_b1 = np.zeros_like(LL_host_b)
    length_b = len(s_LL_host_b)
    s_LL_host_b1[:length_b, :length_b] = np.diag(s_LL_host_b)
    s_LL_host_g1 = np.zeros_like(LL_host_g)
    length_g = len(s_LL_host_g)
    s_LL_host_g1[:length_g, :length_g] = np.diag(s_LL_host_g)
    s_LL_host_r1 = np.zeros_like(LL_host_r)
    length_r = len(s_LL_host_r)
    s_LL_host_r1[:length_r, :length_r] = np.diag(s_LL_host_g)

    s_LL_secret_b1 = np.zeros_like(LL_secret_b)
    length_b = len(s_LL_secret_b)
    s_LL_secret_b1[:length_b, :length_b] = np.diag(s_LL_secret_b)
    s_LL_secret_g1 = np.zeros_like(LL_secret_g)
    length_g = len(s_LL_secret_g)
    s_LL_secret_g1[:length_g, :length_g] = np.diag(s_LL_secret_g)
    s_LL_secret_r1 = np.zeros_like(LL_secret_r)
    length_r = len(s_LL_secret_r)
    s_LL_secret_r1[:length_r, :length_r] = np.diag(s_LL_secret_r)

    alpha = 0.1
    h, w = s_LL_host_b1.shape
    for i in range(h):
        for j in range(w):
            s_LL_host_b1[i, j] = (1-alpha) * s_LL_host_b1[i, j] + alpha * s_LL_secret_b1[i, j]
            s_LL_host_g1[i, j] = (1-alpha) * s_LL_host_g1[i, j] + alpha * s_LL_secret_g1[i, j]
            s_LL_host_r1[i, j] = (1-alpha) * s_LL_host_r1[i, j] + alpha * s_LL_secret_r1[i, j]
    
    # inverse SVD to reconstruct LL component of adversarial example
    LL_b = np.dot(u_LL_host_b, (np.dot(s_LL_host_b1, v_LL_host_b)))
    LL_g = np.dot(u_LL_host_g, (np.dot(s_LL_host_g1, v_LL_host_g)))
    LL_r = np.dot(u_LL_host_r, (np.dot(s_LL_host_r1, v_LL_host_r)))

    # LH component of host image as the one of adversarial example
    LH_b = LH_host_b
    LH_g = LH_host_g
    LH_r = LH_host_r

    # HL component 
    beta = 1.0
    angle = [-5, 5, -10, 10]
    HL_secret_b1 = np.zeros_like(HL_secret_b)
    HL_secret_g1 = np.zeros_like(HL_secret_g)
    HL_secret_r1 = np.zeros_like(HL_secret_r)
    for an in angle:
        HL_secret_b1 += rotate(HL_secret_b, an)
        HL_secret_g1 += rotate(HL_secret_g, an)
        HL_secret_r1 += rotate(HL_secret_r, an)
    HL_secret_b1 /= 4
    HL_secret_g1 /= 4
    HL_secret_r1 /= 4
    HL_b = (1-beta)*HL_host_b + beta*HL_secret_b1
    HL_g = (1-beta)*HL_host_g + beta*HL_secret_g1
    HL_r = (1-beta)*HL_host_r + beta*HL_secret_r1

    # HH component
    HH_b = HH_secret_b
    HH_g = HH_secret_g
    HH_r = HH_secret_r

    # inverse DWT
    coeff_b = LL_b, (HL_b, LH_b, HH_b)
    coeff_g = LL_g, (HL_g, LH_g, HH_g)
    coeff_r = LL_r, (HL_r, LH_r, HH_r)

    stego_b = pywt.idwt2(coeff_b, 'haar')
    stego_g = pywt.idwt2(coeff_g, 'haar')
    stego_r = pywt.idwt2(coeff_r, 'haar')

    stego = cv2.merge([stego_b, stego_g, stego_r])
    imshow_opencv('stego', stego.astype(np.uint8))


    plt.subplot(221)
    plt.imshow(LL_secret, 'gray')
    plt.title('LL')
    plt.subplot(222)
    plt.imshow(LH_secret, 'gray')
    plt.title('LH')
    plt.subplot(223)
    plt.imshow(HL_secret, 'gray')
    plt.title('HL')
    plt.subplot(224)
    plt.imshow(HH_secret, 'gray')
    plt.title('HH')
    plt.show()



if __name__ == '__main__':
    host_img = cv2.imread('images/host.JPEG')
    secret_img = cv2.imread('images/secret.JPEG')
    secret_img = cv2.resize(secret_img, (host_img.shape[1], host_img.shape[0]))
    DWT_SVD(host_img, secret_img)
    cv2.destroyAllWindows()