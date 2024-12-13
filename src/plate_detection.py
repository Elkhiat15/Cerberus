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

    def process_sobel_gradient(self, thresholded_image, ratio):
        """Process Sobel gradient and apply morphological operations"""
        # Sobel x-direction gradient
        sobel_x = cv.Sobel(thresholded_image, cv.CV_64F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        max_gradient_value = np.max(sobel_x)
        sobel_x = 255 * ((sobel_x) / (max_gradient_value))
        sobel_x = sobel_x.astype("uint8")
        
        # Conditional dilation
        gradient_image = sobel_x
        if ratio < 0.0042:
            gradient_image = cv.dilate(sobel_x, None, iterations=2)
        
        # Closing operation
        closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
        closed_image = cv.morphologyEx(gradient_image, cv.MORPH_CLOSE, closing_kernel)
        
        return closed_image

    def refine_image(self, closed_image, ratio):
        """Apply series of morphological operations to refine the image"""
        # Erosion and dilation cycles
        eroded_1 = cv.erode(closed_image, None, iterations=2)
        dilated_1 = cv.dilate(eroded_1, None, iterations=3)
        
        eroded_2 = cv.erode(dilated_1, None, iterations=2)
        dilated_2 = cv.dilate(eroded_2, None, iterations=3)
        
        # Thresholding
        dilated_2[dilated_2 < 140] = 0
        dilated_2[dilated_2 >= 140] = 255
        
        # Additional erosion based on ratio
        eroded_3 = cv.erode(dilated_2, None, iterations=2)
        if ratio >= 0.01:
            erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            eroded_3 = cv.erode(eroded_3, erosion_kernel, iterations=2)
        
        # Final morphological operations
        final_dilated_image = cv.dilate(eroded_3, None, iterations=8)
        vertical_se = cv.getStructuringElement(cv.MORPH_RECT, (5,1))
        final_img = cv.dilate(final_dilated_image, vertical_se, iterations=3)
        
        return final_img


