import cv2


def imshow_opencv(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)


def rotate(img, angle, center=None, scale=1.0):
    h, w = img.shape[:2]
    if center is None:
        center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated