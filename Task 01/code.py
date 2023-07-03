import cv2
import numpy as np

def median_filter(data, filter_size):
    border = filter_size // 2

    median = np.copy(data)

    rows = len(data)
    colums = len(data[0])

    for img_rows in range(rows):
        for img_colums in range(colums):
            temp = []
            for kernel_rows in range(filter_size):
                if img_rows + kernel_rows - border < 0 or img_rows + kernel_rows - border > rows - 1:
                    #for kernel_colums in range(filter_size):
                        next
                else:
                    if img_colums + kernel_rows - border < 0 or img_colums + border > colums - 1:
                        temp.append(0)
                    else:
                        for kernel_colums in range(filter_size):
                            temp.append(data[img_rows + kernel_rows - border][img_colums + kernel_colums - border])

            temp.sort()
            median[img_rows][img_colums] = temp[len(temp) // 2]
    return median


def my_integral(arr_img):
    h, w = len(arr_img), len(arr_img[0])

    """
    Integral image in OpenCV is defined as the sum of pixels in the original image with indices LESS THAN
    those of the integral image, not less than or equal to. Thus, an extra row and colum are needed. In this
    way there is no need to check if my indice is legal.

    For instance, if my rectagle covers the entire image, it may produce an index that falls out of array by 1. 
    That is why the integral image is stored in a size that is by 1x1 larger than the original image.    
    """

    # building the extra row and colum
    arr_v = np.zeros((h, 1))
    arr_h = np.zeros((1, w + 1))

    mat = np.hstack([arr_v, arr_img])
    mat = np.vstack([arr_h, mat])


    h, w = len(mat), len(mat[0])
    integral_image = np.zeros(mat.shape)

    # building integral image
    for y in range(1, h):  # rows
        sum = 0
        for x in range(1, w):  # colums
            sum += mat[y][x]
            integral_image[y][x] = sum + integral_image[y - 1][x]

    # checking if the output is correct
    # img2 = cv2.integral(img)
    # a = np.array_equal(img2, integral_image)


    return integral_image


def main():
    filename = '9_original.png'
    img1 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('main image without noise', img1)

    filename = '9_noise.png'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('main image', img)

    # --------    Filter Application    -------- #
    arr = np.array(img)
    img = median_filter(arr, 3)
    cv2.imshow('filtred image with my median filter', img)

    # img_median_cv = cv2.medianBlur(arr,3)
    # cv2.imshow('filtred image cv2 median filter', img_median_cv)


    # --------    Binary Image    -------- #
    #thresh, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, img_binary = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Image', img_binary)


    # --------    Morphological Operations    -------- #
    # CLOSING
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,2) )
    img_close = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, strel, iterations=2)
    cv2.imshow('Morphological Operations - Closing', img_close)

    # ---- Tries to disconnecet some cells ----
    # median = median_filter(median, 3)
    # cv2.imshow('median filter 2', median)
    # # median = median_filter(median, 3)
    # cv2.imshow('median filter 2', median)

    # open = cv2.morphologyEx(median, cv2.MORPH_OPEN, strel, iterations=1)
    # cv2.imshow('Morphological Operations - Opening', open)

    # strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # open = cv2.morphologyEx(median, cv2.MORPH_OPEN, strel, iterations=2)
    # cv2.imshow('Morphological Operations - Opening', open)

    # img_close2 = cv2.morphologyEx(median, cv2.MORPH_CLOSE, strel, iterations=2)
    # cv2.imshow('Morphological Operations - Closing_2', img_close2)

    # CLOSING_2
    # strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
    # img_close2 = cv2.morphologyEx(median, cv2.MORPH_CLOSE, strel, iterations=4)
    # cv2.imshow('Morphological Operations - Closing 2', img_close2)

    # # OPENING
    # img_open = cv2.morphologyEx(img_close2, cv2.MORPH_OPEN, strel, iterations=4)
    # cv2.imshow('Morphological Operations - Opening', img_open)
    # ----   ----

    eik = img_close

    img_integral = my_integral(eik)

    img_final_rec = np.dstack([img, img, img])   # rgb_img to draw the bounding box and the text

    dok_img = np.dstack([eik, eik, eik])
    img_final_rec1 = dok_img

    # Returns the number of found components in num and an array with the value of each component (e.g. 1,2,...,10) for each pixel
    num, val_pix = cv2.connectedComponents(eik)
    for i in range(1, num):
        counter = 0
        regions = np.zeros(eik.shape, dtype=np.uint8)
        regions[val_pix == i] = 255
        x, y, w, h = cv2.boundingRect(regions)

        # to avoid noise, if the height and width of bounding rectangles are small enough they are considered noise
        if w < 1 or h < 11:
            counter += 1
            continue
        num_regions = i - counter
        print('---- Region Number ', str(num_regions), ": ----")

        color = (0,0,255)   #GBR
        img_final_rec = cv2.rectangle(img_final_rec, (x, y), (x + w, y + h), color, 1)
        img_final_text = cv2.putText(img_final_rec, str(num_regions), (x+10, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        img_final_rec1 = cv2.rectangle(img_final_rec1, (x, y), (x + w, y + h), color, 1)
        img_final_text1 = cv2.putText(img_final_rec1, str(num_regions), (x + 10, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        # Calculating cells area:
        cells_area_box = img_binary[y:y + h, x:x + w]
        cells_area = 0
        for y_i in range(1, cells_area_box.shape[0]):
            for x_i in range(1, cells_area_box.shape[1]):
                if cells_area_box[y_i][x_i] == 255:
                    cells_area += 1
        print("Area (px): ", cells_area)


        # Calculating the bounding box area with the dimensions of width and height
        box_area = w * h
        print("Bounding Box Area (px): ", box_area)

        # Calculating Mean gray-level value in bounding box:
        A = img_integral[y][x]
        B = img_integral[y][x + w]
        C = img_integral[y + h][x]
        D = img_integral[y + h][x + w]
        mglv = (A + D - B - C) / (w * h)
        print("Mean graylevel value in bounding box: ", mglv)


    # ---- RESULTS ----
    cv2.imshow('FINAL BINARY', img_final_text1)
    cv2.imshow('FINAL ORIGINAL', img_final_text)

    # ---- Save images ----
    cv2.imwrite('img_final_binary.png', img_final_text1)
    cv2.imwrite('img_final_original.png', img_final_text)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
