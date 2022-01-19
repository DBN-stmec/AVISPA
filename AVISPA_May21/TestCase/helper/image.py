import numpy
import scipy.misc
import cv2
from skimage import exposure
from PIL import Image


def calculate_brightness(image_np):
    if len(image_np.shape) == 3:
        color_channels_count = image_np.shape[2]
    else:
        color_channels_count = 1
    if color_channels_count == 3: #rgb
        brightness = numpy.average(image_np[2], axis=1, weights=[0.2126, 0.7152, 0.0722])
        brightness = numpy.average(brightness)
    elif color_channels_count == 1: #grayscale
        brightness = numpy.average(image_np[2])
        brightness = numpy.average(brightness)
    else:
        exit("Invalid channel count")
    return brightness


def calculate_sharpness(image_np):
    sharpness_image = scipy.misc.imresize(image_np, (100, 100))
    sharpness = cv2.Laplacian(sharpness_image, cv2.CV_64F).var()
    return int(sharpness)


def histogram_equalization(image):
    cv2.imwrite('tmp2.png', image)
    image = cv2.imread('tmp2.png', 0)
    return cv2.equalizeHist(image)


def adaptive_histogram_equalization(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def contrast_stretching(image, pmin=0, pmax=100):
    min, max = numpy.percentile(image, (1, 99))
    result = exposure.rescale_intensity(image, in_range=(min, max))
    return result


def convert_to_grayscale(image_np):
    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    cvuint8 = cv2.convertScaleAbs(image)
    return cvuint8


def normalize(image_np):
    if image_np.max() != 255:
        return image_np * (255/image_np.max())
    return image_np
