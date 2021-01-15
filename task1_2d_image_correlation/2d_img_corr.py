"""
A simple 2D correlation function.

"""

import cv2
import numpy as np


def correlate2d(image1, image2):
    """
    Return a correlation image of two grayscale images.
    Parameters
    ----------
    image1 : ndarray
        First input grayscale image.
    image2 : ndarray
        Second input grayscale image.
    Returns
    -------
    corr : ndarray
        *.png file with a correlation image.

    """
    image = np.asarray(image1, dtype=np.float32)
    template = np.asarray(image2, dtype=np.float32)

    image_shape, template_shape = \
        np.array(image.shape), np.array(template.shape)

    if np.any(np.less(image_shape, template_shape)):
        raise ValueError("Image1 must be larger than image2.")

    corr_shape = image_shape - template_shape + 1
    corr = np.zeros(corr_shape, dtype=np.float32)

    for i in range(corr_shape[0]):
        for j in range(corr_shape[1]):
            for k, u in enumerate(range(i, i + template_shape[0])):
                for m, v in enumerate(range(j, j + template_shape[1])):
                    corr[i, j] += image[u, v] * template[k, m]
    return corr


if __name__ == "__main__":
    IMAGE_1 = "./images/image.png"
    IMAGE_2 = "./images/template.png"

    img1 = cv2.imread(IMAGE_1)
    img2 = cv2.imread(IMAGE_2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    CORR = correlate2d(gray1, gray2)
    CORR *= 255.0/CORR.max()
    CORR = np.asarray(CORR, dtype=np.uint8)
    print(CORR)

    OUTPUT_IMAGE = "./images/corr.png"
    cv2.imwrite(OUTPUT_IMAGE, CORR)
    # cv2.imshow(OUTPUT_IMAGE, CORR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
