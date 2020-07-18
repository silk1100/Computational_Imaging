import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculateDistance(pt1, pt2):
    if type(pt1) != type(pt2):
        raise ValueError("2 points must be the same type")

    if (type(pt1) == type(3.7))or(type(pt1)==type(np.uint8(5))):
        return pt1-pt2

    else:
        return np.sqrt(np.sum(np.square(pt1-pt2)))


def calculateGauss(num, sigma):
    const = (1.0/(np.sqrt(2*np.pi)*sigma))
    exp = np.exp(-0.5*(num/(sigma*sigma)))
    return exp*const


def imbilat(img, size, sigmaS, sigmaR):
    wr, wc = size
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filt_img = np.zeros_like(img)
    rows, cols = img.shape
    filt_img = np.double(filt_img)
    dimg = np.double(img)

    for r in range(wr//2, rows - wr//2):
        for c in range(wc//2, cols - wc//2):
            rows_range = np.arange(r - wr // 2, r + wr // 2+1)
            cols_range = np.arange(c - wc // 2 , c + wc // 2+1)

            X, Y = np.meshgrid(cols_range, rows_range)

            image_range = dimg[Y, X]

            distRange = np.square(dimg[r, c] - image_range)

            Cgrid = np.square(X - c)
            Rgrid = np.square(Y - r)
            distSpatial = np.sqrt(Cgrid + Rgrid)
            w = calculateGauss(distRange,sigmaR) * calculateGauss(distSpatial, sigmaS)
            k = np.sum(w)

            filt_img[r,c] = np.sum(dimg[Y, X] * w)/k

    return filt_img


def main():
    im = cv2.imread('brain-MRI.jpg')
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    fil = imbilat(img, (9,9), sigmaR=75, sigmaS=75)

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    imfil = cv2.bilateralFilter(src=img, d=9,sigmaColor=75,sigmaSpace=75)
    axes[0].imshow(im)
    axes[1].imshow(fil.astype(np.uint8), cmap='gray')
    axes[2].imshow(imfil.astype(np.uint8), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()

