import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def find_rectangle(contour):
    y, x = [], []
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])
    return [min(y), min(x), max(y), max(x)]

def count_c(t, gray):
    ans = np.zeros((t[0][2] - t[0][0] + 1))
    for i in range(t[0][0], t[0][2]):
        for j in range(t[0][1], t[0][3]):
            if gray[j][i] == 0:
                ans[i-t[0][0]] += 1
    return ans

def count_r(t, gray):
    ans = np.zeros((t[0][3] - t[0][1] + 1))
    for j in range(t[0][1], t[0][3]):
        for i in range(t[0][0], t[0][2]):
            if gray[j][i] == 255:
                ans[j-t[0][1]] += 1
    return ans

def Sobel(crop_img):
    img_sobelX = cv2.Sobel(crop_img, cv2.CV_64F, 1, 0)
    img_sobelY = cv2.Sobel(crop_img, cv2.CV_64F, 0, 1)
    img_sobelX = np.uint8(np.absolute(img_sobelX))
    img_sobelY = np.uint8(np.absolute(img_sobelY))
    img_sobelXY = cv2.bitwise_or(img_sobelX, img_sobelY)
    return img_sobelXY

def find_ang(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # for x in range(0, img_gray.shape[0]-1):
    #     for y in range(0, img_gray.shape[1]-1):
    #         if img_gray[x][y] < 30:
    #             img_gray[x][y] -= 10
    #             if img_gray[x][y] <0:
    #                 img_gray[x][y] = 0
    #         else:
    #             img_gray[x][y] = 225
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), sigmaX=20)
    crop_img = img_gray[20:800, 20:1270]

    img_sobelX = cv2.Sobel(crop_img, cv2.CV_64F, 1, 0)
    img_sobelY = cv2.Sobel(crop_img, cv2.CV_64F, 0, 1)
    img_sobelX = np.uint8(np.absolute(img_sobelX))
    img_sobelY = np.uint8(np.absolute(img_sobelY))
    img_sobelXY = cv2.bitwise_or(img_sobelX, img_sobelY)
    img_sobelXY = img_sobelXY * 4
    ret3, img_th = cv2.threshold(img_sobelXY, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel1 = np.ones((11, 11), np.uint8)
    opening = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel1)
    closing = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel1)
    c2o = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)

    img_show = cv2.cvtColor(c2o, cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(c2o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for c in contours:
        r = find_rectangle(c)
        a = (r[2] - r[0]) * (r[3] - r[1])
        s = (r[2] - r[0]) / (r[3] - r[1])
        blocks.append([r, a, s])
    blocks = sorted(blocks, key=lambda b: b[1])[-1:]
    if blocks[0][0][0] < 300:
        ans = 0
    else:
        ans = 1
    return ans


file_dir = '/Volumes/ADATA HD330/test'
save_file_dir = './output'
file_list = os.listdir(file_dir)
jpg_list = []
for f in file_list:
	if f.endswith(".jpg"):
		jpg_list.append(f)
jpg_list = sorted(jpg_list, key=lambda b: b[:-4])
outtxt = open('410586015.txt', 'w')
print(jpg_list)
for f in jpg_list:
    file_path = os.path.join(file_dir, f)
    output_path = os.path.join(save_file_dir, f)
    print(file_path)
    outtxt.write(f)
    outtxt.write('\n')
    img = cv2.imread(file_path)
    ang =find_ang(img)
    # img = cv2.imread('./sample/003.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), sigmaX=20)

    if ang == 0:
        crop_img = img_gray[680:800, 880:1270]
        # crop_img = img_gray[0:800, 0:1270]
    elif ang == 1:
        crop_img = img_gray[5:115, 5:400]

    # img_sobelXY = Sobel(crop_img)
    img_sobelXY = crop_img
    
    # crop_img = img_sobelXY[650:800, 900:1270]
    # print(crop_img.shape)
    # crop_img = crop_img * 4
    # kernel3 = np.ones((3, 3), np.uint8)
    # img_sobelXY = cv2.morphologyEx(img_sobelXY, cv2.MORPH_OPEN, kernel3)
    # img_sobelXY = img_sobelXY * 2

    # cv2.imshow('crop', crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # img_th = Thresholding(crop_img, eq2_M)
    ret3, img_th = cv2.threshold(img_sobelXY, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(img_sobelXY,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) #换行符号 \
    th3 = cv2.adaptiveThreshold(img_sobelXY,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # ret3, img_th = cv2.threshold(img_sobelXY, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # img_th = cv2.adaptiveThreshold(img_sobelXY, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)

    # for i in range(th2.shape[0]):
    # 	for j in range(th2.shape[1]):
    # 		if th2[i][j] ==255:
    # 			th2[i][j] = 0
    # 		else:
    # 			th2[i][j] = 255

    kernel1 = np.ones((7, 7), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel1)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel2)
    closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel3)
    # cv2.imwrite(output_path, closing)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel1)
    closing2 = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel3)
    c2o = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    # c2o = cv2.morphologyEx(c2o, cv2.MORPH_CLOSE, kernel1)
    c2o2 = cv2.morphologyEx(closing2, cv2.MORPH_OPEN, kernel3)
    # c2o = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)
    # c2oc = cv2.morphologyEx(c2o, cv2.MORPH_CLOSE, kernel1)
    print(c2o.shape)
    c2o = c2o.astype(np.uint8)

    # ret3, c2o = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # img_label = measure.label(c2o)
    img_show = cv2.cvtColor(c2o, cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(c2o, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # cv2.drawContours(img_show, contours, -1, (0, 0, 255), 1)

    blocks = []
    for c in contours:
        r = find_rectangle(c)
        a = (r[2] - r[0]) * (r[3] - r[1])
        s = (r[2] - r[0]) / (r[3] - r[1])
        blocks.append([r, a, s])


    blocks = sorted(blocks, key=lambda b: b[1])[-30:]
    # print(blocks)
    targets = []
    for b in blocks:
        # print(b)
        if b[1] > 1100 and b[0][3] - b[0][1] > 40 and b[1] < 20000:
            # cv2.rectangle(img_show, (b[0][0], b[0][1]), (b[0][2], b[0][3]), (0, 0, 255), 2)
            targets.append([b[0], b[2]])
            # print(b)

    targets = sorted(targets, key=lambda b: b[0][0])
    # print(targets)
    for t in targets:
        if t[1]>0.9:
            count = count_c(t, c2o)
            # print(targets)
            # print(count)
            # x = range(count.shape[0])
            # plt.plot(x, count)
            # plt.show()
            mini = 255
            # mid = math.floor((t[0][2] - t[0][0] + 1)/2)
            # mid1 = math.floor((mid - t[0][0] + 1)/2)
            # mid2 = math.floor((t[0][2] - mid + 1)/2)
            # for x in range(mid-10, mid+10):
            for z in range(20, 40):
                if count[z] < mini:
                    mini = count[z]
                    minx = z
            s = (t[0][2] - t[0][0]-minx) / (t[0][3] - t[0][1])
            targets.append([[t[0][0]+minx, t[0][1], t[0][2], t[0][3]], s])
            t[0][2] = t[0][0]+minx
            t[1] = (t[0][2] - t[0][0]) / (t[0][3] - t[0][1])

    targets = sorted(targets, key=lambda b: b[0][0])
    # print(targets)
    # print(len(targets))

    for t in targets:
        count_col = count_c(t, c2o)
        count_raw = count_r(t, c2o)
        yt = 0
        yb = 0
        xt = 0
        xb = 0
        for x in range(0, count_raw.shape[0]):
            if count_raw[x] < 7:
                yt += 1
            else:
                break
        for y in range(count_raw.shape[0]-1, 0, -1):
            if count_raw[y] < 7:
                yb += 1
            else:
                break
        for x in range(0, count_col.shape[0]):
            if count_col[x] < 7:
                xt += 1
            else:
                break
        for y in range(count_col.shape[0]-1, 0, -1):
            if count_col[y] < 7:
                xb += 1
            else:
                break
        # print(count_raw)
        # print(yb)
        cv2.rectangle(img_show, (t[0][0]+xt, t[0][1]+yt), (t[0][2]-xb, t[0][3]-yb), (0, 0, 255), 2)
        if ang == 0:
            outstr = str(t[0][0]+xt+880) + ' ' + str(t[0][1]+yt+680) + ' ' + str(t[0][2]-xb+880) + ' ' + str(t[0][3]-yb+680)
        elif ang == 1:
            outstr = str(t[0][0]+xt+5) + ' ' + str(t[0][1]+yt+5) + ' ' + str(t[0][2]-xb+5) + ' ' + str(t[0][3]-yb+5)
        outtxt.write(outstr)
        outtxt.write('\n')
        # break
    # cv2.line(img_show, (900, 0), (900, 800), (0, 0, 255), 1)
    # cv2.line(img_show, (1270, 0), (1270, 800), (0, 0, 255), 1)
    # cv2.line(img_show, (0, 650), (1280, 650), (0, 0, 255), 1)
    # cv2.line(img_show, (0, 770), (1280, 770), (0, 0, 255), 1)

    # print(img.shape)
    # print(img_gray.shape)
    output_path = os.path.join(save_file_dir, f)
    cv2.imwrite(output_path, img_show)
    # cv2.imwrite(output_path, img_label)

    # cv2.imshow('img_gray', img_gray)
    # cv2.imshow('img_label', img_label)
    # cv2.imshow('crop', crop_img)
    # cv2.imshow('img_canny', img_canny)
    # cv2.imshow('img_sobel', img_sobelXY)
    # cv2.imshow('img_th', img_th)
    # cv2.imshow('th2', th2)
    # cv2.imshow('th3', th3)
    # cv2.imshow('opening', opening)
    # cv2.imshow('closing', closing)
    # cv2.imshow('closing2', closing2)
    # cv2.imshow('c2o', c2o)
    # cv2.imshow('c2o2', c2o2)
    # cv2.imshow('img_show', img_show)
    # cv2.imshow('o2c', o2c)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()