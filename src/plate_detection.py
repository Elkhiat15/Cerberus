import cv2 as cv
import imutils
import numpy as np
from contours_utils import * 


class LicensePlateDetector:
    def __init__(self):
        """Initialize the License Plate Detector"""
        pass

    def preprocess_image(self, img):
        """Resize and crop the image"""
        image_shape= img.shape
        img = imutils.resize(img, width=1000, height=1000 * image_shape[0] // image_shape[1])
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = gray[gray.shape[0]*2//5:gray.shape[0], :]
        gray = cv.GaussianBlur(gray, (3,3), 0)
        
        img = img[img.shape[0]*2//5:img.shape[0], :]
        return img, gray

    def apply_black_hat_morphology(self, gray):
        """Apply black hat morphological operation"""
        rectangle_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,5))
        black_hat_image = cv.morphologyEx(gray, cv.MORPH_BLACKHAT,  rectangle_kernel)
        return black_hat_image

    def threshold_and_analyze_image(self, black_hat_image):
        """Threshold the image and calculate pixel ratio"""
        thresholded_image = black_hat_image.copy()
        thresholded_image[thresholded_image < 50] = 0
        thresholded_image[thresholded_image >= 50] = 255
        
        white_pixel_count = np.count_nonzero(thresholded_image == 255)
        black_pixel_count = np.count_nonzero(thresholded_image == 0)
        ratio = round(white_pixel_count / (black_pixel_count + white_pixel_count), 4)
        
        return thresholded_image, ratio

