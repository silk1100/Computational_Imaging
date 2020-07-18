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


def calculateUfUbTheta(pdf, t, u):
    theta = 0.0001
    ut = 0.0
    for i in range(t+1):
        theta += pdf[i]
        ut += i*pdf[i]
    uf = ut/theta
    ub = (u-ut)/(1-theta)

    return uf, ub, theta


def calculateSfSb(pdf, t, theta, uf, ub):
    numSf = 0
    for i in range(t+1):
        numSf += ((i-uf)*(i-uf))*pdf[i]
    Sf = numSf/theta

    numSb = 0
    for i in range(t+1, len(pdf)):
        numSb += ((i-ub)*(i-ub))*pdf[i]
    Sb = numSb/(1-theta)

    return Sf, Sb


def calculateSwSb(u, uf,ub, Sf, Sb, theta):
    Sw = (theta*Sf) + ((1-theta)*Sb)
    Sb = (theta*(uf-u)*(uf-u)) + ((1-theta)*(u-ub)*(u-ub))
    return Sw, Sb


def otsuThresh_toptim_SWs_SBs(img):
    u = np.mean(img)
    h = calculatePDF(img)
    sws = np.zeros_like(h)
    sbs = np.zeros_like(h)
    ind = 0
    for t in np.arange(256):
        uf, ub, theta = calculateUfUbTheta(h, t, u)
        Sf, Sb = calculateSfSb(h, t, theta, uf, ub)
        Sw, Sb = calculateSwSb(u, uf, ub, Sf, Sb, theta)
        sws[ind] = Sw
        sbs[ind] = Sb
        ind += 1

    toptim = np.argmin(sws)

    return toptim, sws, sbs


def main():
    img = cv2.imread('Chest_CT_IM.png')
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    toptim, sws, sbs = otsuThresh_toptim_SWs_SBs(img)
    bin_img = np.zeros_like(img)
    bin_img[np.where(img >= toptim)] = 255
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    axes[2].plot(sws, label='Sigma within', c='r')
    axes[2].plot(sbs, label='Sigma Between')
    axes[2].axvline(toptim, c='y', label='Optimum threshold')
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[1].imshow(bin_img, cmap='gray')
    axes[1].set_title('Otsu binarized image')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()