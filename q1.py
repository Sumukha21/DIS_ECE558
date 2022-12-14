import numpy as np


def pad_image(image, pad_type, padding_width, padding_height):
    """
    Function to perform the padding to the image given the padding parameters
    :param image: The image for which the padding needs to be performed
           type: A 2D single or multichannel array
    :param pad_type: The type of padding that needs to be performed. The following types are supported:
                     a) zero: Fills the padding area with zeros
                     b) wrap_around: The image is treated as a circular cylinder and the values from the end of the
                                     image continue to the start of the image
                     c) copy_edge: Copies the edge values from the boundary columns and rows for all the padding area
                     d) reflect: Reflects the values at the boundaries to the padding area
               type: string
    :param padding_width: The number of columns which needs to be added for padding on each side
                    type: int
    :param padding_height: The number of rows which needs to be added for padding on each side
                     type: int
    :return: returns the padded image
    """
    padded_image = np.zeros(
        (image.shape[0] + (2 * padding_height), image.shape[1] + (2 * padding_width), image.shape[-1]))
    padded_image[padding_height: padded_image.shape[0] - padding_height,
                 padding_width: padded_image.shape[1] - padding_width, :] = image
    if pad_type == "zero":
        return padded_image
    elif pad_type == "wrap_around":  # circular
        # padded_image2 = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width,
        #                                    cv2.BORDER_WRAP)
        padded_image[padding_height: padded_image.shape[0] - padding_height,
        padding_width: padded_image.shape[1] - padding_width, :] = image
        if padding_height > 0:
            padded_image[0: padding_height, padding_width: image.shape[1] + padding_width, :] = \
                image[-padding_height:, :, :]
            padded_image[image.shape[0] + padding_height:, padding_width: image.shape[1] + padding_width, :] = \
                image[: padding_height, :, :]
        if padding_width > 0:
            padded_image[padding_height: image.shape[0] + padding_height, 0: padding_width, :] = \
                image[:, -padding_width:, :]
            padded_image[padding_height: image.shape[0] + padding_height, image.shape[1] + padding_width:, :] =\
                image[:, : padding_width, :]
        if padding_height > 0 and padding_width > 0:
            padded_image[0: padding_height, 0: padding_width, :] = image[image.shape[0] - padding_height:,
                                                                         image.shape[1] - padding_width:, :]
            padded_image[-padding_height:, -padding_width:, :] = image[0: padding_height, 0: padding_width, :]
            padded_image[0: padding_height, -padding_width:, :] = image[-padding_height:, 0: padding_width, :]
            padded_image[-padding_height:, 0: padding_width:, :] = image[0: padding_height:, -padding_width:, :]
    elif pad_type == "copy_edge":
        # padded_image2 = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_REPLICATE)
        if padding_width > 0:
            for i in range(padding_width):
                padded_image[padding_height: image.shape[0] + padding_height, i, :] = image[0:, 0, :]
                padded_image[padding_height: image.shape[0] + padding_height, -i - 1, :] = image[0:, -1, :]
        if padding_height > 0:
            for i in range(padding_height):
                padded_image[i, padding_width: image.shape[1] + padding_width, :] = image[0, 0:, :]
                padded_image[-i - 1, padding_width: image.shape[1] + padding_width, :] = image[-1, 0:, :]
        if padding_height > 0 and padding_width > 0:
            padded_image[0: padding_height, 0: padding_width, :] = image[0, 0, :]
            padded_image[-padding_height:, -padding_width:, :] = image[-1, -1, :]
            padded_image[-padding_height:, 0: padding_width, :] = image[-1, 0, :]
            padded_image[0: padding_height, -padding_width:, :] = image[0, -1, :]
    elif pad_type == "reflect":
        # padded_image2 = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width,
        #                                    cv2.BORDER_REFLECT)
        if padding_height > 0:
            for i in range(image.shape[-1]):
                padded_image[0: padding_height, padding_width: image.shape[1] + padding_width, i] = np.flip(
                    image[0: padding_height, :, i], 0)
                padded_image[-padding_height:, padding_width: image.shape[1] + padding_width, i] = np.flip(
                    image[-padding_height:, :, i], 0)
        if padding_width > 0:
            for i in range(image.shape[-1]):
                padded_image[padding_height: image.shape[0] + padding_height, 0: padding_width, i] = np.flip(
                    image[:, 0: padding_width, i], 1)
                padded_image[padding_height: image.shape[0] + padding_height, -padding_width:, i] = np.flip(
                    image[:, -padding_width:, i], 1)
        if padding_height > 0 and padding_width > 0:
            for i in range(image.shape[-1]):
                padded_image[0: padding_height, 0: padding_width, i] = np.flip(image[0: padding_height, 0: padding_width, i], 0)
                padded_image[-padding_height:, 0: padding_width, i] = np.flip(
                    image[-padding_height, 0: padding_width, i], 0)
                padded_image[0: padding_height, -padding_width:, i] = np.flip(
                    image[0: padding_height, -padding_width:, i], 0)
                padded_image[-padding_height:, -padding_width:, i] = np.flip(
                    image[-padding_height:, -padding_width:, i], 0)
    else:
        raise NotImplementedError("Provided padding type is not supported."
                                  " Please check the doc string to see the supported types")
    return padded_image


def conv2d(f, w, pad, stride=1):
    """
    Computes the cross correlation between the input image f and the provided filter w
    :param f: The input image which can be either a gray or color image
           type: A 2D single or multichannel array
    :param w: The provided kernel with which the convolution needs to be performed
           type: A 2D ndarray
    :param pad: Indicates the padding type that needs to be performed for the image
           type: string
    :param stride: Indicates the stride of the kernel when convolving with the image
           type: int
    :return: Returns the convolved image
    """
    input_width = f.shape[1]
    input_height = f.shape[0]
    if len(f.shape) == 2:
        f = f[:, :, np.newaxis]
    input_depth = f.shape[-1]
    kernel_width = w.shape[1]
    kernel_height = w.shape[0]
    padded_image = pad_image(f, pad, kernel_width // 2, kernel_height // 2)
    output_width = int(((input_width - kernel_width + (2 * (kernel_width // 2))) / stride) + 1)
    output_height = int(((input_height - kernel_height + (2 * (kernel_height // 2))) / stride) + 1)
    output_image = np.zeros((output_height, output_width, input_depth))
    w = np.flip(w)
    for i in range(0, input_width, stride):
        for j in range(0, input_height, stride):
            for k in range(input_depth):
                output_image[j, i, k] = np.sum(padded_image[j: j + kernel_height, i: i + kernel_width, k] * w)
    output_image_cropped = np.squeeze(output_image[: input_height, : input_width, :])
    return output_image_cropped


if __name__ == "__main1__":
    """
    Testing the padding
    """
    import cv2
    import os

    img_path = r".\ECE_558\Project_2\inputs\lena.png"
    s_path = r".\ECE_558\Project_2\q1_padding"
    if not os.path.exists(s_path):
        os.mkdir(s_path)
    img = cv2.imread(img_path)
    pad_types = ["zero", "copy_edge", "reflect", "wrap_around"]
    pad_width = 25
    pad_height = 25
    for pad_type in pad_types:
        padded_img = pad_image(img, pad_type, pad_width, pad_height)
        cv2.imwrite(os.path.join(s_path, "lena_padding_%s.png" % pad_type), padded_img)


if __name__ == "__main2__":
    """
    Performing 2D convolution with set of inputs and a list filters
    """
    import cv2
    import os

    f_path = r".\ECE_558\Project_2\inputs"
    s_path = r".\ECE_558\Project_2\q1"
    image_inputs = ["wolves.png", "lena.png"]
    filters = ["box", "f_der_row", "f_der_col", "prewitt_z", "prewitt_y", "sobel_x", "sobel_y", "roberts_x", "robter_y"]
    padding_types = ["zero", "copy_edge", "reflect", "wrap_around"]
    image_types = [0, -1]
    for image_input in image_inputs:
        image_path = os.path.join(f_path, image_input)
        image_name = image_input.split(".png")[0]
        save_path = os.path.join(s_path, image_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for image_type in image_types:
            image = cv2.imread(image_path, image_type)
            for filter in filters:
                for padding_type in padding_types:
                    if filter == "box":
                        w_ = np.ones((3, 3)) / 9
                    elif filter == "f_der_row":
                        w_ = np.asarray([[-1, 1]])
                    elif filter == "f_der_col":
                        w_ = np.asarray([[-1], [1]])
                    elif filter == "prewitt_z":
                        w_ = np.asarray([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                    elif filter == "prewitt_y":
                        w_ = np.asarray([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                    elif filter == "sobel_x":
                        w_ = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                    elif filter == "sobel_y":
                        w_ = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                    elif filter == "roberts_x":
                        w_ = np.asarray([[0, 1], [-1, 0]])
                    elif filter == "roberts_y":
                        w_ = np.asarray([[1, 0], [0, -1]])
                    convolved_image = conv2d(image, w_, padding_type)
                    save_path_i = os.path.join(save_path, "color_space_%s_filter_%s_padding_type_%s.png"
                                               % (["gray", "rgb"][image_type], filter, padding_type))
                    print(save_path_i)
                    cv2.imwrite(save_path_i, convolved_image)


if __name__ == "__main3__":
    """
    Checking the impulse response
    """
    import os
    import matplotlib.pyplot as plt

    s_path = r".\ECE_558\Project_2\q1"
    img_size = [1024, 1024]
    impulse_input = np.zeros((1024, 1024))
    impulse_input[511, 511] = 1
    filter_types = ["box_filter", "prewitt_y"]
    for filter_type in filter_types:
        if filter_type == "prewitt_y":
            example_kernel = np.asarray([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        else:
            example_kernel = np.ones((3, 3)) / 9
        padding = "zero"
        result = conv2d(impulse_input, example_kernel, padding)
        plt.title("%s_impulse_response.png" % filter_type)
        plt.imshow(result, cmap="gray")
        plt.savefig(os.path.join(s_path, "%s_impulse_response.png" % filter_type))
        print("")
