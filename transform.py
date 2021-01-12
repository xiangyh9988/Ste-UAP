import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from utils.imutils import *
import kornia

def process_on_channel(host, secret):
    ''' DWT on host and secret '''
    LL_h, (LH_h, HL_h, HH_h) = pywt.dwt2(host, 'haar')
    LL_s, (LH_s, HL_s, HH_s) = pywt.dwt2(secret, 'haar')

    ''' fusion of LL component '''
    # SVD on LL components
    u_LL_h, s_LL_h, v_LL_h = np.linalg.svd(LL_h, full_matrices=False, compute_uv=True)
    u_LL_s, s_LL_s, v_LL_s = np.linalg.svd(LL_s, full_matrices=False, compute_uv=True)
    # fusion
    alpha = 0.1
    s_LL = (1-alpha)*s_LL_h + alpha*s_LL_s
    LL = np.dot(u_LL_h*s_LL, v_LL_h)

    ''' LH component from host image '''
    LH = LH_h

    ''' fusion of HL component '''
    beta = 1.0
    angle = [-5, 5, -10, 10]
    HL_s_rotated = np.mean(np.array([rotate(HL_s, an) for an in angle]), axis=0)
    HL = (1-beta)*HL_h + beta*HL_s_rotated

    ''' HH component from secret image '''
    HH = HH_s

    ''' inverse DWT to reconstruct stego image '''
    coeffs = LL_h, (LH_s, HL_s, HH_s)
    stego = pywt.idwt2(coeffs, 'haar')

    return stego

def DWT_SVD(host_img, secret_img):
    # imshow_opencv('host', host_img)
    # imshow_opencv('secret', secret_img)
    
    ''' split image to channels '''
    host_r, host_g, host_b = cv2.split(host_img)
    secret_r, secret_g, secret_b = cv2.split(secret_img)
    ''' process image on channel separately '''
    stego_r = process_on_channel(host_r, secret_r)
    stego_g = process_on_channel(host_g, secret_g)
    stego_b = process_on_channel(host_b, secret_b)

    ''' merge channels to reconstruct BGR stego image '''
    stego = cv2.merge([stego_r, stego_g, stego_b])
    if host_img.shape[0] % 2 == 1 and stego.shape[0] % 2 == 0:
        stego = cv2.resize(stego, (host_img.shape[1], host_img.shape[0]))
    # normalize to [0, 1]
    stego = (stego - stego.min())/(stego.max()-stego.min())
    # convert to [0, 255] int to save
    stego = np.uint8(stego*255)
    # imshow_opencv('int stego', stego)
    # cv2.imwrite('stego.png', stego[:, :, ::-1])
    return stego

# if __name__ == '__main__':
#     host_img = cv2.imread('images/host.JPEG')
#     secret_img = cv2.imread('images/secret.JPEG')
#     ''' resize secret image to host image's size '''
#     secret_img = cv2.resize(secret_img, (host_img.shape[1], host_img.shape[0]))
#     stego_img = DWT_SVD(host_img, secret_img)
#     cv2.destroyAllWindows()