import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from utils.imutils import *

def DWT_SVD(host_img, secret_img):
    imshow_opencv('host', host_img)
    imshow_opencv('secret', secret_img)

    ''' perform DWT to decompose images to 4 components '''
    # DWT on host x1
    host_b, host_g, host_r = cv2.split(host_img)
    # dwt: LL, HL, LH, HH
    # HL: high in h, low in v
    # LH: low in h, high in v
    LL_host_b, (HL_host_b, LH_host_b, HH_host_b) = pywt.dwt2(host_b, 'haar')
    LL_host_g, (HL_host_g, LH_host_g, HH_host_g) = pywt.dwt2(host_g, 'haar')
    LL_host_r, (HL_host_r, LH_host_r, HH_host_r) = pywt.dwt2(host_r, 'haar')

    # DWT on secret x2
    secret_b, secret_g, secret_r = cv2.split(secret_img)
    LL_secret_b, (HL_secret_b, LH_secret_b, HH_secret_b) = pywt.dwt2(secret_b, 'haar')
    LL_secret_g, (HL_secret_g, LH_secret_g, HH_secret_g) = pywt.dwt2(secret_g, 'haar')
    LL_secret_r, (HL_secret_r, LH_secret_r, HH_secret_r) = pywt.dwt2(secret_r, 'haar')

    ''' fusion of LL component '''
    # SVD on host and secret （not full matrices)
    u_LL_host_b, s_LL_host_b, v_LL_host_b = np.linalg.svd(LL_host_b, full_matrices=0, compute_uv=1)
    u_LL_host_g, s_LL_host_g, v_LL_host_g = np.linalg.svd(LL_host_g, full_matrices=0, compute_uv=1)
    u_LL_host_r, s_LL_host_r, v_LL_host_r = np.linalg.svd(LL_host_r, full_matrices=0, compute_uv=1)
    u_LL_secret_b, s_LL_secret_b, v_LL_secret_b = np.linalg.svd(LL_secret_b, full_matrices=0, compute_uv=1)
    u_LL_secret_g, s_LL_secret_g, v_LL_secret_g = np.linalg.svd(LL_secret_g, full_matrices=0, compute_uv=1)
    u_LL_secret_r, s_LL_secret_r, v_LL_secret_r = np.linalg.svd(LL_secret_r, full_matrices=0, compute_uv=1)

    # fusion 
    alpha = 0.1
    s_LL_b1 = (1-alpha)*s_LL_host_b + alpha*s_LL_secret_b
    s_LL_g1 = (1-alpha)*s_LL_host_g + alpha*s_LL_secret_g
    s_LL_r1 = (1-alpha)*s_LL_host_r + alpha*s_LL_secret_r
    LL_b = np.dot(u_LL_host_b*s_LL_b1, v_LL_host_b)
    LL_g = np.dot(u_LL_host_g*s_LL_g1, v_LL_host_g)
    LL_r = np.dot(u_LL_host_r*s_LL_r1, v_LL_host_r)

    ''' LH component of host image as the one of adversarial example '''
    LH_b = LH_host_b
    LH_g = LH_host_g
    LH_r = LH_host_r

    ''' HL component fusion '''
    beta = 1
    angle = [-5, 5, -10, 10]
    HL_secret_b_rotated = np.mean(np.array([rotate(HL_secret_b, an) for an in angle]), axis=0)
    HL_secret_g_rotated = np.mean(np.array([rotate(HL_secret_g, an) for an in angle]), axis=0)
    HL_secret_r_rotated = np.mean(np.array([rotate(HL_secret_r, an) for an in angle]), axis=0)
    HL_b = (1-beta)*HL_host_b + beta*HL_secret_b_rotated
    HL_g = (1-beta)*HL_host_g + beta*HL_secret_g_rotated
    HL_r = (1-beta)*HL_host_r + beta*HL_secret_r_rotated

    ''' HH component '''
    HH_b = HH_secret_b
    HH_g = HH_secret_g
    HH_r = HH_secret_r

    ''' inverse DWT to reconstruct image '''
    coeff_b = LL_b, (HL_b, LH_b, HH_b)
    coeff_g = LL_g, (HL_g, LH_g, HH_g)
    coeff_r = LL_r, (HL_r, LH_r, HH_r)

    stego_b = pywt.idwt2(coeff_b, 'haar')
    stego_g = pywt.idwt2(coeff_g, 'haar')
    stego_r = pywt.idwt2(coeff_r, 'haar')

    stego = cv2.merge([stego_b, stego_g, stego_r])
    # normalize to visualize
    stego = (stego - stego.min())/(stego.max()-stego.min())
    imshow_opencv('stego', stego)
    # convert to [0, 255] to save
    stego = np.uint8(stego*255)
    cv2.imwrite('stego.png', stego)


if __name__ == '__main__':
    host_img = cv2.imread('images/host.JPEG')
    secret_img = cv2.imread('images/secret.JPEG')
    secret_img = cv2.resize(secret_img, (host_img.shape[1], host_img.shape[0]))
    DWT_SVD(host_img, secret_img)
    # DWT_SVD_gray(host_img, secret_img)
    cv2.destroyAllWindows()