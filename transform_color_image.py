import os
import argparse
import logging
import cv2
import numpy as np
from skimage.transform import radon

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')

def main(
    imageFilepath,
    outputDirectory
):
    logging.info("transform_color_image.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the image
    original_img = cv2.imread(imageFilepath)
    # Split the B,G,R channels
    bgr_channels = cv2.split(original_img)

    # Create an array of linearly spaced angles
    theta = np.linspace(0., 180., max(original_img.shape), endpoint=False)

    # Radon transform of each channel, independently
    sinogram_blue = radon(bgr_channels[0], theta=theta, circle=False)
    sinogram_green = radon(bgr_channels[1], theta=theta, circle=False)
    sinogram_red = radon(bgr_channels[2], theta=theta, circle=False)
    logging.info("sinogram_blue.shape = {}".format(sinogram_blue.shape))

    # Rescale the sinograms with the largest value
    sinogram_blue, sinogram_green, sinogram_red, scaling_factor = Rescale(sinogram_blue, sinogram_green, sinogram_red)

    # Stack the sinograms into a 3 channel color image
    sinogram = cv2.merge([sinogram_blue, sinogram_green, sinogram_red])
    cv2.namedWindow("Sinogram", cv2.WINDOW_NORMAL)
    cv2.imshow("Sinogram", sinogram)
    cv2.waitKey(0)

    cv2.imwrite(os.path.join(outputDirectory, "sinogram.png"), sinogram)

def Rescale(sinogram_blue, sinogram_green, sinogram_red):
    max_value = np.max(sinogram_blue)
    max_value = max(max_value, np.max(sinogram_green))
    max_value = max(max_value, np.max(sinogram_red))
    sinogram_blue = sinogram_blue * 255/max_value
    sinogram_green = sinogram_green * 255/max_value
    sinogram_red = sinogram_red * 255/max_value
    return sinogram_blue.astype(np.uint8), sinogram_green.astype(np.uint8), sinogram_red.astype(np.uint8), 255/max_value



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageFilepath", help="The filepath to the image. Default: './images/plants_small_color.jpg'", default='./images/plants_small_color.jpg')
    parser.add_argument("--outputDirectory", help="The output directory. Default: './outputs'", default='./outputs')
    args = parser.parse_args()

    main(
        args.imageFilepath,
        args.outputDirectory
    )