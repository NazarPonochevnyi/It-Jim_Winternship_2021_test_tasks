"""
Function that detects (finds xmin, xmax, ymin, ymax of) black spots on images
and visualize results by surrounding each spot (blob) with a red rectangle.

"""

import os
import warnings

import cv2


def detect_black_spots(image_folder,
                       threshold=127,
                       window=5,
                       rectangle_color=(0, 0, 255),
                       rectangle_border_size=2):
    """
    Return black spot's coordinates and result images with a rectangle.
    Parameters
    ----------
    image_folder : string
        Path to folder with input images.
    threshold : int, optional
        Threshold filter, that used to detect black spots.
        By default set to 127.
    window : int, optional
        Window filter, that used to remove some short chunks of noise.
        By default set to 5.
    rectangle_color : tuple of int, optional
        Color of a rectangle as a tuple of integers in BGR.
        By default set to red.
    rectangle_border_size : int, optional
        Border size of the rectangle.
        By default set to 2.
    Returns
    -------
    output1 : list of dicts
        Result data in format:
        [{'file': SR1.png, 'coords': [left,right,top,bottom]}, ...]
    output2 : list of ndarrays
        *.png images as ndarrays with a visualization of results.

    """
    output1, output2 = [], []
    for img_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, img_name)

        # open image and transform to grayscale
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find y_min, y_max
        y_min, y_max, y_1, y_2 = -1, -1, -1, -1
        for i, row in enumerate(gray):
            if min(row) < threshold:
                if y_1 == -1:
                    y_1, y_2 = i, i
                else:
                    if i - y_2 <= window:
                        y_2 = i
                    else:
                        if y_2 - y_1 > y_max - y_min:
                            y_min, y_max = y_1, y_2
                        y_1, y_2 = i, i
        if y_2 - y_1 > y_max - y_min:
            y_min, y_max = y_1, y_2
        if y_min == -1 or y_max == -1:
            y_min, y_max = 0, img.shape[0]

        # find x_min, x_max
        x_min, x_max, x_1, x_2 = -1, -1, -1, -1
        for i, column in enumerate(gray.T):
            if min(column) < threshold:
                if x_1 == -1:
                    x_1, x_2 = i, i
                else:
                    if i - x_2 <= window:
                        x_2 = i
                    else:
                        if x_2 - x_1 > x_max - x_min:
                            x_min, x_max = x_1, x_2
                        x_1, x_2 = i, i
        if x_2 - x_1 > x_max - x_min:
            x_min, x_max = x_1, x_2
        if x_min == -1 or x_max == -1:
            warnings.warn(
                f"Can't find black spot on '{img_name}' image. "
                "Try to increase threshold or window values."
            )
            x_min, x_max = 0, img.shape[1]

        # draw rectangle and save the result
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), rectangle_color, \
            rectangle_border_size)
        output1.append({"file": img_name, \
                       "coords": [x_min, x_max, y_min, y_max]})
        output2.append(img)
    return output1, output2


if __name__ == "__main__":
    IMAGE_FOLDER = "./images/"
    OUT1, OUT2 = detect_black_spots(IMAGE_FOLDER)
    print(OUT1)

    # show or save output2 images
    OUTPUT_FOLDER = "./result/"
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    for j, image in enumerate(OUT2):
        name = "result_" + OUT1[j]["file"]
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, name), image)
        # cv2.imshow(name, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
