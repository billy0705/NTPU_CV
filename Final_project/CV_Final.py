import cv2
import numpy as np

def erosion(img, kernel):
    img_01 = img / 255
    img_output = np.zeros((img_01.shape), np.uint8)
    ky = (int)((kernel.shape[0] - 1) / 2)
    kx = (int)((kernel.shape[1] - 1) / 2)
    count = 0
    for y in range(img_01.shape[0]):
        for x in range(img_01.shape[1]):
            if x < kx or y < ky or y > img_01.shape[0] - ky - 1 or x > img_01.shape[1] - kx - 1:
                img_output[y][x] = 0
            else:
                com = img_01[y - ky: y + ky + 1, x - kx: x + kx + 1]
                com = com.astype('uint8')
                # print(y - ky, y + ky + 1, com.shape)
                if (com == kernel).all() == True:
                    count += 1
                    img_output[y][x] = 1
                else:
                    img_output[y][x] = 0

    img_output = img_output * 255
    return img_output

def dilation(img, kernel):
    img_01 = img / 255
    img_output = np.zeros((img_01.shape), np.uint8)
    ky = (int)((kernel.shape[0] - 1) / 2)
    kx = (int)((kernel.shape[1] - 1) / 2)
    count = 0
    for y in range(img_01.shape[0]):
        for x in range(img_01.shape[1]):
            if img_01[y][x] == 1:
                count += 1
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[1]):
                        a = y-ky+i
                        b = x-kx+j
                        if a<0:
                            a = 0
                        elif a>img_01.shape[0] - 1:
                            a = img_01.shape[0] - 1
                        else:
                            a = a
                        if b<0:
                            b = 0
                        elif b>img_01.shape[1] - 1:
                            b = img_01.shape[1] - 1
                        else:
                            b = b
                        img_output[a][b] = 1
    # print(count)
    img_output = img_output * 255
    return img_output

def opening(img, kernel):
    img_er = erosion(img, kernel)
    img_output = dilation(img_er, kernel)
    return img_output

def closing(img, kernel):
    img_di = dilation(img, kernel)
    img_output = erosion(img_di, kernel)
    return img_output


img = cv2.imread('./sample/005.jpg')
img = cv2.imread('./456.png')
# img = cv2.imread('./test1.jpg')
print(img.shape)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), sigmaX=20)
# img_th = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
ret3, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_th = 255 - img_th
print(img_th.shape)

kernel = np.ones((13, 13), np.uint8)
print(kernel.shape)
img_ero = erosion(img_th, kernel)
img_dil = dilation(img_th, kernel)
img_op = opening(img_th, kernel)
img_clo = closing(img_th, kernel)


cv2.imshow('img_gray', img_gray)
cv2.imshow('img_th', img_th)
cv2.imshow('img_ero', img_ero)
cv2.imshow('img_dil', img_dil)
cv2.imshow('img_op', img_op)
cv2.imshow('img_clo', img_clo)
cv2.waitKey(0)
cv2.destroyAllWindows()