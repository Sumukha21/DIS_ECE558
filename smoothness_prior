import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt


def color_space_transformation(img, color_space="rgb"):
    """
    Given the image array and target color space, color space transformation is done and resulting image is returned
    :param img: An image array which needs to be transformed to the required color space
           type: ndarray
    :param color_space: The color space to which the image needs to be transformed
           type: "string"
           valid values: set("rgb", "hsv", "gray", "lab)
    :return: The transformed image
    """
    if color_space == "rgb":
        return img
    elif color_space == "gray":
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)
        return img[:, :, np.newaxis]
    elif color_space == "hsv":
        return cv2.cvtColor(np.asarray(img, np.float32), cv2.COLOR_RGB2HSV)
    elif color_space == "lab":
        return cv2.cvtColor(np.asarray(img, np.float32), cv2.COLOR_RGB2LAB)
    else:
        raise NotImplementedError("Provided option for color space '%s' is not a valid one. "
                                  "Please look at the doc string to see the valid options" % color_space)


def compute_image_smoothness(img, neighbor="d00", img_color_space="rgb"):
    """
    Computes the list of square of intensity difference for every valid pair of a pixel (x, y) and its selected
    neighbor
    param img: A 2D/3D image array
           type: ndarray
    param neighbor: The neighbor to be selected to compute the pixel intensity difference with. The neighbor will
    belong to the set of 8 neighbors of each pixel in the image. Following are the valid neighbors:
            ________________________________________________        ______________________
            | (x - 1, y - 1) | (x, y - 1) | (x + 1, y - 1) |        | d00 |   d01  | d02 |
            ------------------------------------------------        ----------------------
            |  (x - 1, y)    |   (x, y)   |   (x + 1, y)   |   =>   | d10 | (x, y) | d12 |
            ------------------------------------------------        ----------------------
            | (x - 1, y + 1) | (x, y + 1) | (x + 1, y + 1) |        | d20 |  d21  |  d22 |
            ------------------------------------------------        ----------------------
    return: List of difference in intensities
    """
    shift_dict = {"d00": [1, 0, 1, 0], "d01": [1, 0, 0, 0], "d02": [1, 0, 0, 1], "d10": [0, 0, 1, 0],
                  "d12": [0, 0, 0, 1], "d20": [1, 0, 0, 1], "d21": [0, 1, 0, 0], "d22": [0, 1, 0, 1]}
    img = color_space_transformation(img, img_color_space.lower())
    img = np.asarray(img, dtype=np.int32)
    if shift_dict.get(neighbor):
        shift = shift_dict[neighbor]
    else:
        raise NotImplementedError("Provided option for neighbor '%s' is not a valid one."
                                  " Please look at the doc string to see the valid options" % neighbor)
    img_1 = img[shift[0]: img.shape[0] - shift[1], shift[2]:img.shape[1] - shift[3], :]
    img_2 = img[shift[1]: img.shape[0] - shift[0], shift[3]:img.shape[1] - shift[2], :]
    difference = (img_1 - img_2) ** 2
    if len(img.shape) > 2:
        difference_merged = np.zeros(difference.shape[0:2])
        for i in range(img.shape[-1]):
            difference_merged += difference[:, :, i]
        return list(np.asarray(difference_merged.reshape(-1), np.int32))
    return list(np.asarray(difference.reshape(-1), np.int32))


def compute_histogram(img_path, save_folder, neighbor_list, color_spaces_list):
    """
    Given the image path and list of neighbors and list of color spaces, the histogram of differences of intensities are
    computed and the resulting histogram is saved in the provided folder
    param img_path: Path to the image for which the smoothness needs to be computed
           type: string
    param save_folder: Path to the folder where the histograms need to be saved
           type: string
    param neighbor_list: List of neighbor types for which the histogram of intensity differences need to be computed
           type: List of string values
    param color_spaces_list: List of neighbor types for which the histogram of intensity differences need to be
    computed
           type: List of string values
    """
    neighbor_to_coordinate_mapping = {"d00": "(x - 1, y - 1)", "d01": "(x, y - 1)", "d02": "(x + 1, y - 1)",
                                      "d10": "(x - 1, y)", "d12": "(x + 1, y)",
                                      "d20": "(x - 1, y + 1)", "d21": "(x, y + 1)", "d22": "(x + 1, y + 1)"}
    img = cv2.imread(img_path)
    img = np.flip(img, 2)
    run_time = []
    for neighbor in neighbor_list:
        print("Working on the following neighbor: ", neighbor_to_coordinate_mapping[neighbor])
        for color_space in color_spaces_list:
            start = time.time()
            print("     Working on the following color space: ", color_space)
            intensity_differences = compute_image_smoothness(img, neighbor, color_space)
            plt.figure(figsize=(10, 10))
            plt.hist(intensity_differences)
            plt.xlabel("Square Intensity Difference", fontsize=14)
            plt.ylabel("Number of pixel pairs", fontsize=14)
            plt.title("Color Space is %s and Neighbor selected is %s" % (color_space,
                                                                         neighbor_to_coordinate_mapping[neighbor]),
                                                                         fontsize=16)
            plt.savefig(os.path.join(save_folder, "histogram_%s_neighbor_%s.png" % (color_space,
                                                                                    neighbor_to_coordinate_mapping[neighbor])))
            plt.close()
            run_time.append(time.time() - start)
    print("  Time taken: %.2f seconds" % np.mean(run_time))


if __name__ == "__main__":
    image_path = "./wolves.png"
    save_path = "./ECE_558/Project_1/Project_1_Option_0_solution/part_1_results"
    # neighbors = ["d00", "d01", "d02", "d10", "d12", "d20", "d21", "d22"]
    neighbors = ["d12"]
    color_spaces = ["rgb", "gray", "hsv", "lab"]
    # color_spaces = ["lab"]
    compute_histogram(image_path, save_path, neighbors, color_spaces)
