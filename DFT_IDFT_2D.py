import numpy as np
import cv2


def normalize_image(upper_limit, image):
    """
    Normalizes the image and scales it to the provided upper limit
    :param upper_limit: The value by which the normalized image needs to be scaled
                  type: int/float or any real number
    :param image: The image which needs to be normalized
            type: ndarray
    :return: The normalized image
    """
    return np.asarray(((image - np.min(image)) / (np.max(image) - np.min(img))) * upper_limit, np.float32)


def denormalize_image(image, image_max, image_min):
    """
    Denormalizes an image given its initial maximum and minimum values
    :param image: The image which needs to be denormalized
            type: ndarray
    :param image_max: The initial maximum value of the image which can be used to scale the image back
                type: integer/float or any real number
    :param image_min: The initial minimum value of the image which can be used to scale the image back
                type: integer/float or any real number
    :return: The denormalized image
    """
    image = (image * (image_max - image_min)) + image_min
    return image


def compute_dft2d_naive(image):
    """
    Naive implementation of DFT2D using the DFT2D definition
    :param image: The image for which the DFT2D transformation needs to be computed
    :return: The frequency domain transformed image, spectrum and corresponding phase
    """
    height, width = image.shape
    dft2d = np.zeros((height, width), dtype=complex)
    for u in range(height):
        for v in range(width):
            u_v_sum = 0.0
            for u_ in range(height):
                for v_ in range(width):
                    basis = np.exp(- 2j * np.pi * ((u * u_) / height + (v * v_) / width))
                    u_v_sum += image[u_, v_] * basis
            dft2d[u, v] = u_v_sum
    spectrum = np.sqrt(np.square(dft2d.real) + np.square(dft2d.imag))
    phase_angle = np.arctan(dft2d.imag/dft2d.real)
    return dft2d, spectrum, phase_angle


def dft2d_fft_based(image):
    """
    Implementation of DFT2D using the 1D FFT using the separable property of DFT2D
    :param image: The image for which the DFT2D transformation needs to be computed
    :return: The frequency domain transformed image, spectrum and corresponding phase
    """
    # image = normalize_image(1, image)
    image = np.asarray(image, np.complex)
    for i in range(image.shape[0]):
        image[i, :] = np.fft.fft(image[i, :])
    for i in range(image.shape[1]):
        image[:, i] = np.fft.fft(image[:, i])
    spectrum = np.sqrt(np.square(image.real) + np.square(image.imag))
    phase_angle = np.arctan2(image.imag, image.real)
    return image, spectrum, phase_angle


def idft2d_fft_based(image):
    """
    Implementation of IDFT2D using the 1D FFT
    :param image: The image for which the DFT2D transformation needs to be computed
    :return: The frequency domain transformed image, spectrum and corresponding phase
    """
    for i in range(image.shape[0]):
        image[i, :] = np.array(1j, np.complex) * np.conjugate(image[i, :])
        image[i, :] = np.fft.fft(image[i, :])
        image[i, :] = np.array(1j, np.complex) * np.conjugate(image[i, :])
    image /= image.shape[0]
    for i in range(image.shape[1]):
        image[:, i] = np.array(1j, np.complex) * np.conjugate(image[:, i])
        image[:, i] = np.fft.fft(image[:, i])
        image[:, i] = np.array(1j, np.complex) * np.conjugate(image[:, i])
    image /= image.shape[1]
    return image


if __name__ == "__main2__":
    """
    Testing the fft1D based DTF2D with naive DFT2D implementation
    """
    img_path = r".\ECE_558\Project_2\inputs\lena.png"
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (40, 40))
    img = normalize_image(1, img)
    dft2d_image = compute_dft2d_naive(img)
    dft2d_image2 = dft2d_fft_based(img)
    print("")


if __name__ == "__main3__":
    """
    Testing the dft2d and idft2d using fft1D
    """
    import os
    import matplotlib.pyplot as plt

    input_folder = r".\ECE_558\Project_2\inputs"
    input_image_name = ["lena.png", "wolves.png"]
    save_folder = r".\ECE_558\Project_2\q2"
    for input_name in input_image_name:
        img = cv2.imread(os.path.join(input_folder, input_name), 0)
        img_max = np.max(img)
        img_min = np.min(img)
        dft2d_image, dft_spectrum, dft_phase = dft2d_fft_based(img)
        dft2d_in_built = np.fft.fft2(img)
        idft2d_image = idft2d_fft_based(dft2d_image)
        img_reconstructed = np.asarray(np.round(denormalize_image(np.abs(idft2d_image), img_max, img_min)), np.uint8)
        reconstruction_error = img_reconstructed - img
        plt.title("%s input image" % input_name)
        plt.imshow(img, cmap="gray")
        plt.savefig(os.path.join(save_folder, "%s_input_image.png" % input_name))
        plt.cla()
        plt.title("%s DFT2D spectrum" % input_name)
        plt.imshow(np.log(1 + dft_spectrum), cmap="gray")
        plt.savefig(os.path.join(save_folder, "%s_dft_spectrum.png" % input_name))
        plt.cla()
        plt.title("%s DFT2D phase" % input_name)
        plt.imshow(np.fft.fftshift(dft_phase), cmap="gray")
        plt.savefig(os.path.join(save_folder, "%s_dft_phase.png" % input_name))
        plt.cla()
        plt.title("%s IDFT2D output" % input_name)
        plt.imshow(img_reconstructed, cmap="gray")
        plt.savefig(os.path.join(save_folder, "%s_idft_denormalized.png" % input_name))
        plt.cla()
        plt.title("%s Reconstruction error" % input_name)
        plt.imshow(reconstruction_error, cmap="gray")
        plt.savefig(os.path.join(save_folder, "%s_idft_reconstruction_error.png" % input_name))
        plt.cla()


if __name__ == "__main__":
    img = np.ones((3, 3))
    padded1 = np.zeros((5, 5))
    padded1[0: 3, 0: 3] = img
    padded2 = np.zeros((5, 5))
    padded2[1: 4, 1: 4] = img
    dft1 = dft2d_fft_based(padded1)
    dft2 = dft2d_fft_based(padded2)
    print("")
