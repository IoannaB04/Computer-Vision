import numpy as np
import cv2 as cv

" Import images in gray scale "

# img1 = cv.imread('rio-01.png', cv.IMREAD_GRAYSCALE)
# img2 = cv.imread('rio-02.png', cv.IMREAD_GRAYSCALE)
# img3 = cv.imread('rio-03.png', cv.IMREAD_GRAYSCALE)
# img4 = cv.imread('rio-04.png', cv.IMREAD_GRAYSCALE)

img1 = cv.imread('mine01.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('mine02.jpg', cv.IMREAD_GRAYSCALE)
img3 = cv.imread('mine03.jpg', cv.IMREAD_GRAYSCALE)
img4 = cv.imread('mine04.jpg', cv.IMREAD_GRAYSCALE)


# algo = "sift"
algo = "surf"

" CODE HERE"
# Show images
# cv.namedWindow('image 01', cv.WINDOW_NORMAL)
# cv.imshow('image 01', img1)
#
# cv.namedWindow('image 02', cv.WINDOW_NORMAL)
# cv.imshow('image 02', img2)
#
# cv.namedWindow('image 03', cv.WINDOW_NORMAL)
# cv.imshow('image 03', img3)
#
# cv.namedWindow('image 04', cv.WINDOW_NORMAL)
# cv.imshow('image 04', img4)


print ("Starting ...")

def match2(d1, d2):         # LAB CODE
    n1 = d1.shape[0]

    matches = []
    for i in range(n1):
        fv = d1[i, :]
        diff = d2 - fv
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        i2 = np.argmin(distances)
        mindist2 = distances[i2]

        # change the value of the minimum distance to infinity in order to tack the second minimum distance next time
        distances[i2] = np.inf

        i3 = np.argmin(distances)
        mindist3 = distances[i3]

        # GOOD MATCHING
        if mindist2 / mindist3 < 0.5:
            matches.append(cv.DMatch(i, i2, mindist2))

    return matches

def Keypoints_Detectors (imgA, imgB,algo):
    if algo == "sift":
        type = cv.xfeatures2d_SIFT.create(1000)
    elif algo == "surf":
        type = cv.xfeatures2d_SURF.create(1000)

    print('Finding key points and descriptors ...')

    # KEY POINTS + DESCRIPTOR FOR IMAGE A
    kpA = type.detect(imgA)
    dA  = type.compute(imgA, kpA)

    # KEY POINTS + DESCRIPTOR FOR IMAGE B
    kpB = type.detect(imgB)
    dB  = type.compute(imgB, kpB)

    print('Finding true matches ...')

    matchesAB = match2(dA[1], dB[1])
    matchesBA = match2(dB[1], dA[1])

    true_matches = [m for m in matchesAB for n in matchesBA if m.distance == n.distance]

    img_ptA = []
    img_ptB = []
    for x in true_matches:
        img_ptA.append( kpA[x.queryIdx].pt )
        img_ptB.append( kpB[x.trainIdx].pt )

    img_ptA = np.array(img_ptA)
    img_ptB = np.array(img_ptB)

    print ('Drawing Matches ...')
    dimg = cv.drawMatches(imgA, dA[0], imgB, dB[0], true_matches, None)
    cv.namedWindow('Crosschecking', cv.WINDOW_NORMAL)
    cv.imshow('Crosschecking', dimg)

    return img_ptA, img_ptB

def stiching(imgA, imgB, algo):
    " HOMOGRAPHY " # Βρίσκει πώς πρέπει να μετατραπει η πρώτη για να "ταιριάξει" με τη δευτερη
    img_ptA, img_ptB = Keypoints_Detectors(imgA, imgB, algo)
    H, mask = cv.findHomography(img_ptB, img_ptA, cv.RANSAC)    # source, destination, We use RANSAC algorithm

    # Wrap destination image to source based on homography
    result = cv.warpPerspective(imgB, H, (imgA.shape[1]+1000, imgA.shape[0]+1000))  # source, H, (destination)
    result[0: imgA.shape[0], 0: imgA.shape[1]] = imgA

    print("Stiching and cropping ...")

    stiched_img = crop(result)

    return stiched_img

def crop(img):
    column_top_delete = column_delete_top(img)
    # cv.namedWindow('after column top delete', cv.WINDOW_NORMAL)
    # cv.imshow('after column top delete', column_top_delete)
    # cv.waitKey(0)
    rows_delete = row_delete(column_top_delete)
    # cv.namedWindow('after row delete', cv.WINDOW_NORMAL)
    # cv.imshow('after row delete', rows_delete)
    # cv.waitKey(0)
    cropped_img = column_delete_bottom(rows_delete)
    # cv.namedWindow('after columns bottom delete', cv.WINDOW_NORMAL)
    # cv.imshow('after columns bottom delete', cropped_img)
    # cv.waitKey(0)
    return cropped_img

def row_delete(img):
    rows = len(img)
    columns = len(img[0])
    print('~~~~ Delete rows')
    for i in range(rows-1):
        sum = 0
        for j in range (columns):
            sum = sum+img[i][j]

        if sum < 10:
            for a in range (rows-i):
                deleted_row = (rows-1)-a
                img = np.delete(img, deleted_row, 0)
            return img

    return img

def column_delete_top(img):
    columns = len(img[0])
    print('~~~~ Delete upper columns')
    for j in range(columns):
        if img[0][j]== 0:
            # print(j)
            for a in range(columns-j):
                deleted_column = (columns-1)-a
                img = np.delete(img, deleted_column, 1)
            return img
    return img

def column_delete_bottom(img):
    rows = len(img)
    columns = len(img[0])
    print('~~~~ Delete down columns')
    if (img[rows - 1][int(columns/2)]  > 0):
        print("BREAK")
        return img

    for j in range(columns):
        if img[rows-1][j] != 0:
            for a in range(columns-j):
                deleted_column = (columns-1)-a
                img = np.delete(img, deleted_column, 1)
            return img
    return img

def panorama (img1, img2, img3, img4, algo):
    print('---- First stiching ----')
    first_stiching = stiching(img1, img2, algo)
    cv.namedWindow('First stiching', cv.WINDOW_NORMAL)
    cv.imshow('First stiching', first_stiching)
    cv.waitKey(0)
    cv.imwrite('stiching_first.png', first_stiching)

    print('---- Second stiching ----')
    second_stiching = stiching(img3, img4, algo)
    cv.namedWindow('Second stiching', cv.WINDOW_NORMAL)
    cv.imshow('Second stiching', second_stiching)
    cv.waitKey(0)
    cv.imwrite('stiching_second.png', second_stiching)

    print('---- Final stiching ----')
    third_stiching = stiching(first_stiching, second_stiching, algo)
    cv.namedWindow('Final stiching', cv.WINDOW_NORMAL)
    cv.imshow('Final stiching', third_stiching)
    cv.waitKey(0)
    cv.imwrite('stiching_third.png', third_stiching)

    return third_stiching

teliko = panorama (img1, img2, img3, img4, algo)
cv.imwrite('teliko.png', teliko)

cv.waitKey(0)
cv.destroyAllWindows()
