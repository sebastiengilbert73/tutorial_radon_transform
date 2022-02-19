import os
import argparse
import logging
import cv2
import numpy as np
from skimage.transform import radon
import ast
import math
from skimage.transform import radon, rescale, iradon_sart, iradon

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')

def main(
        imageSizeHW,
        outputDirectory
):
    logging.info("simulate_bga.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    number_of_angles = max(imageSizeHW)

    # Simulate BGA density
    bga_density_img = GenerateBGADensity(imageSizeHW)

    # Create an image of line integrals
    logging.info("Computing the Radon transform the naïve way...")
    number_of_rhos = math.ceil(math.sqrt(imageSizeHW[0]**2 + imageSizeHW[1]**2))  # The image diagonal defines the highest number of rho's
    rho_max = round(math.sqrt(imageSizeHW[0]**2 + imageSizeHW[1]**2)/2)
    center = (imageSizeHW[1]/2, imageSizeHW[0]/2)
    naive_sinogram = NaiveRadonTransform(bga_density_img, number_of_angles, number_of_rhos, rho_max, center)
    cv2.imwrite(os.path.join(outputDirectory, "simulateBga_main_naiveSinogram.png"), naive_sinogram.astype(np.uint8))

    logging.info("Computing the Radon transform the efficient way, with skimage...")
    thetas = np.linspace(0., 180., number_of_angles, endpoint=False)
    skimage_sinogram = radon(bga_density_img, theta=thetas, circle=False)
    cv2.imwrite(os.path.join(outputDirectory, "simulateBga_main_skimageSinogram.png"), skimage_sinogram)

    reconstruction_iradon = iradon(naive_sinogram, theta=thetas, filter_name='ramp')
    cv2.imwrite(os.path.join(outputDirectory, "simulateBga_main_reconstructionIradon.png"),
                (np.clip(255 * reconstruction_iradon, 0, 255)).astype(np.uint8))


def GenerateBGADensity(image_sizeHW, grid_size=(16, 16), ball_diameter=8, defect_location=(98, 142)):
    density_img = np.zeros(image_sizeHW, dtype=np.uint8)
    stepXY = (image_sizeHW[1]/(grid_size[1] + 1), image_sizeHW[0]/(grid_size[0] + 1))
    for grid_row in range(1, grid_size[0] + 1):
        for grid_col in range(1, grid_size[1] + 1):
            center = (round(stepXY[0] * grid_col), round(stepXY[1] * grid_row))
            cv2.circle(density_img, center, ball_diameter//2, 255, thickness=-1)
    # Create a satellite solder ball
    cv2.circle(density_img, defect_location, ball_diameter//4, 255, thickness=-1)
    return density_img.astype(np.float32)/255

def LineIntegral(density_arr, rho, theta, rho_max, center):
    line_integral = 0.0
    anchor = ( rho * math.sin(theta),  rho * math.cos(theta))
    running_rho = -rho_max
    while running_rho < rho_max:
        running_point = (anchor[0] + running_rho * math.cos(theta), anchor[1] - running_rho * math.sin(theta))
        translated_running_point = ( round(center[0] + running_point[0]), round(center[1] + running_point[1]))
        if translated_running_point[0] >= 0 and translated_running_point[0] < density_arr.shape[1] and \
                translated_running_point[1] >= 0 and translated_running_point[1] < density_arr.shape[0]:
            line_integral += density_arr[translated_running_point[1], translated_running_point[0]]
        running_rho += 1.0
    return line_integral

def NaiveRadonTransform(density_arr, number_of_angles, number_of_rhos, rho_max, center):
    line_integrals_arr = np.zeros((number_of_rhos, number_of_angles), dtype=np.float32)
    for thetaNdx in range(number_of_angles):
        theta = -math.pi/2 + thetaNdx * math.pi/number_of_angles
        for rhoNdx in range(number_of_rhos):
            rho = -rho_max + rhoNdx
            line_integral = LineIntegral(density_arr, rho, theta, rho_max, center)
            line_integrals_arr[number_of_rhos - rhoNdx - 1, thetaNdx] = line_integral  # Negative rho is at bottom, positive rho is at top
    return line_integrals_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageSizeHW", help="The image size (Height, Width). Default: '(256, 256)'",
                        default='(256, 256)')
    parser.add_argument("--outputDirectory", help="The output directory. Default: './outputs_simulate_bga'", default='./outputs_simulate_bga')
    args = parser.parse_args()

    imageSizeHW = ast.literal_eval(args.imageSizeHW)

    main(
        imageSizeHW,
        args.outputDirectory,
        #args.numberOfAngles
    )