import cv2
import matplotlib.pyplot as plt

def imshow_opencv(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)


def show_frequency_component(title, FCs, mode=None):
    FC = cv2.merge(FCs)
    if mode == 'LL':
        # normalize
        FC = (FC - FC.min()) / (FC.max() - FC.min())
    # cv2.imshow(title, cv2.resize(FC, (FC.shape[1]*2, FC.shape[0]*2)))
    cv2.imshow(title, FC)
    cv2.waitKey(0)

def rotate(img, angle, center=None, scale=1.0):
    h, w = img.shape[:2]
    if center is None:
        center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated