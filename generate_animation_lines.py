import os
import argparse
import logging
import cv2
import numpy as np
from skimage.transform import radon
import ast
import imageio
import transform_color_image
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')

def main(
        imageSizeHW,
        outputDirectory,
        numberOfSteps
):
    logging.info("generate_animation_lines.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Create an array of linearly spaced angles
    theta = np.linspace(0., 180., max(imageSizeHW), endpoint=False)

    # Compute the sinogram size with a dummy image
    dummy_img = np.zeros(imageSizeHW, dtype=np.uint8)
    dummy_sinogram = radon(dummy_img, theta=theta, circle=False)
    logging.info("dummy_sinogram.shape = {}".format(dummy_sinogram.shape))
    sinogram_size = dummy_sinogram.shape

    images_list = []
    for stepNdx in range(numberOfSteps):
        print (".", end="", flush=1)
        stacked_image = np.zeros((imageSizeHW[0] + sinogram_size[0], imageSizeHW[1], 3), dtype=np.uint8)
        # Draw the image features
        image = np.zeros((imageSizeHW[0], imageSizeHW[1], 3), dtype=np.uint8)
        # A sweeping blue vertical line
        sweeping_column = round(stepNdx/numberOfSteps * imageSizeHW[1])
        cv2.line(image, (sweeping_column - 20, 0), (sweeping_column + 20, imageSizeHW[0] - 1), (255, 0, 0), thickness=3)
        # A rotating green line
        center = (imageSizeHW[1]//2, imageSizeHW[0]//2)
        beta = np.pi * stepNdx/numberOfSteps
        radius = max(imageSizeHW)//2
        p1 = (round(center[0] + radius * math.cos(beta)), round(center[1] - radius * math.sin(beta)))
        p2 = (round(center[0] - radius * math.cos(beta)), round(center[1] + radius * math.sin(beta)))
        cv2.line(image, p1, p2, (0, 255, 0), thickness=3)
        # An orbiting red line
        d_orbit = max(imageSizeHW)//3
        #p1 = (round(center[0] + d_orbit * math.cos(beta) - radius * math.sin(beta)), round(center[1] - d_orbit * math.sin(beta) - radius * math.sin(beta) ))
        p1 = (round(center[0] + d_orbit * math.cos(2*beta) + radius * math.cos(math.pi/2 - 2*beta)),
              round(center[1] - d_orbit * math.sin(2*beta) + radius * math.sin(math.pi/2 - 2*beta)))
        p2 = (round(center[0] + d_orbit * math.cos(2*beta) - radius * math.cos(math.pi/2 - 2*beta)),
              round(center[1] - d_orbit * math.sin(2*beta) - radius * math.sin(math.pi/2 - 2*beta) ))
        cv2.line(image, p1, p2, (0, 0, 255), thickness=3)

        # Split the color channels
        bgr_channels = cv2.split(image)

        # Radon transform of each channel, independently
        sinogram_blue = radon(bgr_channels[0], theta=theta, circle=False)
        sinogram_green = radon(bgr_channels[1], theta=theta, circle=False)
        sinogram_red = radon(bgr_channels[2], theta=theta, circle=False)
        # Rescale the sinograms with the largest value
        sinogram_blue, sinogram_green, sinogram_red, scaling_factor = transform_color_image.Rescale(sinogram_blue, sinogram_green,
                                                                              sinogram_red)
        # Stack the sinograms into a 3 channel color image
        sinogram = cv2.merge([sinogram_blue, sinogram_green, sinogram_red])

        stacked_image[0: imageSizeHW[0], :, :] = image
        stacked_image[imageSizeHW[0]:, :, :] = sinogram

        stacked_image_rgb = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2RGB)  # imageio expects RGB images
        images_list.append(stacked_image_rgb)
    print()

    animated_gif_filepath = os.path.join(outputDirectory, "lines_radon.gif")
    imageio.mimsave(animated_gif_filepath, images_list, format='GIF', fps=60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageSizeHW", help="The image size (Height, Width). Default: '(100, 100)'",
                        default='(100, 100)')
    parser.add_argument("--outputDirectory", help="The output directory. Default: './outputs'", default='./outputs')
    parser.add_argument("--numberOfSteps", help="The number of steps. Default: 64", type=int, default=64)
    args = parser.parse_args()

    imageSizeHW = ast.literal_eval(args.imageSizeHW)
    main(
        imageSizeHW,
        args.outputDirectory,
        args.numberOfSteps
    )