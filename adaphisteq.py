import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculatePDF(img):
    """
    Calculate normalized histogram of an image
    :param img: numpy.array - image
    :return h: numpy.array - 1D histogram in case of gray image, 3D histogram in case of color img
    """
    if len(img.shape) == 3:
        c1img = img[:,:,0]
        c2img = img[:,:,1]
        c3img = img[:,:,2]
        # c1MaxVal = np.max(c1img);c2MaxVal = np.max(c2img);c3MaxVal = np.max(c3img)
        # c1MinVal = np.min(c1img);c2MinVal = np.min(c2img);c3MinVal = np.min(c3img)
        # overAllMax = np.max([c1MaxVal, c2MaxVal, c3MaxVal])
        # overAllMin = np.min([c1MinVal, c2MinVal, c3MinVal])

        overAllMax = 255
        overAllMin = 0

        h = np.zeros((overAllMax-overAllMin+1,3))
        for i in range(overAllMin, overAllMax+1):
            h[i, 0] = len(np.where(c1img==i)[0])
            h[i, 1] = len(np.where(c2img == i)[0])
            h[i, 2] = len(np.where(c3img == i)[0])

        h[:, 0] = h[:, 0]/np.sum(h[:,0])
        h[:, 1] = h[:, 1] / np.sum(h[:, 1])
        h[:, 2] = h[:, 2] / np.sum(h[:, 2])
    else:
        # Supposed to be 256, however I am handling if the image has more gray levels
        # It is worth noting that histogram bars are separated with only one gray level
        # maxVal = np.max(img)
        # minVal = np.min(img)
        maxVal = 255
        minVal = 0
        h = np.zeros(maxVal-minVal+1)
        for i in range(minVal, maxVal+1):
            places = np.where(img == i)
            h[i] = len(places[0])
        h = h/np.sum(h)

    return h


def calculateCDF(pdf):
    """
    Calculate the cumulative distribution function using a 1D or 3D probability density function
    :param pdf: numpy.array - probability density function
    :return c: numpy.array - probability cumulative function with the same dimension as pdf
    """
    cdf = np.zeros_like(pdf)
    dims = len(pdf.shape)
    if dims == 1:
        for i in range(len(pdf)):
            cdf[i] = np.sum(pdf[:i+1])
    elif dims == 2:
        for j in range(pdf.shape[1]):
            p = pdf[:, j]
            for i in range(len(p)):
                cdf[i, j] = np.sum(p[:i+1])

    else:
        raise ValueError("PDF is passed with unrecognized dimensions")

    return cdf


def isRCIncluded(listofregions, rctuple):
    """
    Check if a point (r,c) belongs to any regions in the list of regions
    :param listofregions: [[]] - list of lists. Represents lists of regions within the image
                                - and each region list contains the points (tuples) that comprises
                                - that region
    :param rctuple: tuple - containing the row, and column of a point in the image
    :return : bool - to
    """
    # If we don't have any regions included then return False
    if len(listofregions) <= 0:
        return False

    # For every region
    for reg in listofregions:
        # For every point within this region
        for r, c in reg:
            # If this is the same point, then return True we included it before
            if (r ==rctuple[0]) and (c==rctuple[1]):
                return True
    # If we looped over all all points in all regions and we didn't find our rc point
    # Then it has never been included before
    return False


def regionGrowing(img, seeds=[(0,0)], thresh = -1, regionMaxSize=-1, regionMinSize=-1):
    """

    :param img: numpy.array - Original image
    :param seed: tuple - (rows, cols) of starting point
    :param Threshold: numpy.uint8 - maximum difference in gray level allowed to include a pixel to region
    :param regionMaxSize:
    :param regionMinSize:
    :return:
    """

    list_of_regions = list()
    excluded_pixels = list()
    rows, cols = img.shape
    # for r in range(rows):
    #     for c in range(cols):
    for seed in seeds:
        r, c = seed
        if isRCIncluded(list_of_regions,(r, c)):
            continue

        gray_rc = img[r, c]
        region_rc = list()
        s = list()
        s.append((r,c))
        while (len(s)>0) and (len(region_rc) < regionMaxSize):
            cr, cc = s[0]
            neighbors = [(cr-1, cc-1), (cr-1, cc), (cr-1, cc+1),
                         (cr, cc-1), (cr, cc+1),
                         (cr+1, cc-1), (cr+1, cc), (cr+1, cc+1)]
            for neighbor in neighbors:
                nr, nc = neighbor
                if (neighbor in s) or (neighbor in region_rc) or \
                    isRCIncluded(list_of_regions,(nr, nc)):
                    continue
                if (nr < 0) or (nr >= rows):
                    continue
                if (nc < 0) or (nc >= cols):
                    continue
                if abs(float(img[nr, nc]) - float(gray_rc)) < thresh:
                    s.append(neighbor)

            region_rc.append(s[0])
            s.pop(0)

        if len(region_rc) >= regionMinSize:
            list_of_regions.append(region_rc)
        else:
            excluded_pixels.extend(region_rc)

    seg_img = np.zeros_like(img)
    nRegions = len(list_of_regions)
    gray_levels = np.linspace(50, 255, nRegions)

    for ind, region in enumerate(list_of_regions):
        for r, c in region:
            seg_img[r, c] = np.uint8(gray_levels[ind])

    return list_of_regions, seg_img


def imEqualize(img, listofRegions=[]):
    """
    Apply histogram equalization on img. Note if the image is a colored image then the
    equalization is applied on each color separately
    :param img: numpy.array -  image to be equalized
    :return eqImg: numpy.array - histogram equalized image
    """
    eqImg = np.copy(img)
    if len(listofRegions) <= 0:
        h = calculatePDF(img)
        cdf = calculateCDF(h)
        dims = len(cdf.shape)

        if dims == 1:
            for i in range(len(cdf)):
                ng = cdf[i]
                ris, cis = np.where(img == i)
                nGrayValue = np.uint8(ng * 255)
                for r, c in zip(ris, cis):
                    eqImg[r, c] = nGrayValue
        else:
            for j in range(cdf.shape[1]):
                c_j = cdf[:, j]
                for i in range(len(c_j)):
                    ng = c_j[i]
                    ris, cis = np.where(img[:,:,j] == i)
                    nGrayValue = np.uint8(ng * 255)
                    for r, c in zip(ris, cis):
                        eqImg[r, c, j] = nGrayValue

    else:
        for region in listofRegions:
            himg = np.ones_like(img).astype(np.double)
            himg *= -5
            cnt = 0
            for r, c in region:
                himg[r, c] = img[r, c]
                cnt += 1
            h = np.zeros(256)
            c = np.zeros(256)
            for k in range(256):
                h[k] = len(np.where(himg==k)[0])
                c[k] = np.sum(h)

            for ind, cv in enumerate(c):
                eqImg[np.where(himg)==ind] = np.uint8(cv*255)

    return eqImg


def main():
    img = cv2.imread('homeless.PNG')
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    axes[0].imshow(grayimg, cmap='gray')
    axes[0].set_title('Original image')
    # histo, bins = np.histogram(grayimg.reshape(-1, 1), bins=249, density=True)
    # h = calculatePDF(grayimg)
    # c = calculateCDF(h)
    # plt.plot(h, c='r', marker='x')
    # plt.plot(c)
    # plt.show()
    # plt.show()
    # eqImg = imEqualize(img)
    rows, cols = grayimg.shape
    list_of_regions, seg_img = regionGrowing(grayimg,
                                             seeds=[(173, 82),(104,124),(47, 94)],
                                             thresh=30,
                                             regionMinSize=200,
                                             regionMaxSize=int(2*float(rows) * float(cols)/3))
                                             # regionMinSize=float(rows) * float(cols) / 10.0,
                                             # regionMaxSize=2*float(rows) * float(cols)/3)

    #
    #
    #
    eqimg = imEqualize(grayimg, list_of_regions)
    axes[1].imshow(eqimg, cmap='gray')
    axes[1].set_title('Equalized image')
    axes[2].imshow(seg_img, cmap='gray')
    axes[2].set_title('Segmented image')
    plt.show()


if __name__ == '__main__':
    main()
