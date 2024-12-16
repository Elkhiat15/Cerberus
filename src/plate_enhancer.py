import cv2 as cv
import imutils
import numpy as np
from skimage import measure

class LicensePlateEnhancer:
    def __init__(self):
        """Initialize the License Plate Enhancer"""
        self.plate_img = None
        self.preprocessed_image = None
        self.white_mask = None
        self.black_mask = None

    def _is_validate_input(self, plate_img):
        """ Validate input image """
        return not np.all(plate_img == 0)

    def _convert_and_resize_image(self, plate_img):
        """ Convert image to HSV and resize """
        # Convert to HSV and extract value channel
        hsv_value_channel = cv.split(cv.cvtColor(plate_img, cv.COLOR_BGR2HSV))[2]
        
        # Invert and resize image
        preprocessed_image = cv.bitwise_not(hsv_value_channel)
        plate_img = imutils.resize(plate_img, width=200)
        preprocessed_image = imutils.resize(preprocessed_image, width=200)
        
        return preprocessed_image, plate_img

    def _apply_thresholding(self, thresholded_image):
        """ Apply binary thresholding to the image """
        # Binary thresholding
        thresholded_image = thresholded_image.copy()
        thresholded_image[thresholded_image < 120] = 0
        thresholded_image[thresholded_image >= 120] = 255
        
        # Dilation
        dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,4))
        thresholded_image = cv.dilate(thresholded_image, dilation_kernel, iterations=1)
        
        return thresholded_image

    def _process_connected_components(self, preprocessed_image, plate_img):
        """ Process connected components and filter based on characteristics """
        # Label connected components
        labels = measure.label(preprocessed_image, background=0)
        
        # Initialize images
        self.black_mask = np.zeros(preprocessed_image.shape, dtype='uint8')
        self.white_mask = np.zeros(preprocessed_image.shape, dtype='uint8')
        
        # Process each labeled component
        for label in np.unique(labels):
            if label == 0:
                continue
            
            # Create label mask
            current_label_mask = np.zeros(preprocessed_image.shape, dtype='uint8')
            current_label_mask[labels == label] = 255
            
            # Find contours
            label_contours = cv.findContours(current_label_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            label_contours = label_contours[1] if imutils.is_cv3() else label_contours[0]
            
            # Process contours
            if len(label_contours) > 0:
                self._filter_and_process_contour(label_contours, current_label_mask, plate_img)
        
        return self.black_mask

    def _filter_and_process_contour(self, label_contours, current_label_mask, plate_img):
        """ Filter contours based on specific criteria """
        # Get the largest contour
        largest_contour = max(label_contours, key=cv.contourArea)
        
        # Calculate contour characteristics
        (_, _, contour_width, contour_height) = cv.boundingRect(largest_contour)
        area = cv.contourArea(largest_contour)
        height_ratio = contour_height / float(plate_img.shape[0])
        width_ratio = contour_width / float(plate_img.shape[1])
        aspect_ratio = contour_width / float(contour_height)
        contour_solidity = cv.contourArea(largest_contour) / float(contour_width * contour_height)
        
        # Update white image
        self.white_mask = cv.bitwise_or(self.white_mask, current_label_mask)
        
        # Apply filtering criteria
        if (area >= 15 and area < 600 and 
            height_ratio < 0.9 and 
            width_ratio < 0.2 and 
            aspect_ratio < 2 and 
            contour_solidity > 0.2 and 
            contour_solidity < 0.8):
            self.black_mask = cv.bitwise_or(self.black_mask, current_label_mask)

    def enhance_plate(self, plate_img):
        """ Main enhancement method """
        # Validate input
        if not self._is_validate_input(plate_img):
            return []
        
        # Convert and resize image
        self.preprocessed_image, self.plate_img = self._convert_and_resize_image(plate_img)
        
        # Apply thresholding
        self.preprocessed_image = self._apply_thresholding(self.preprocessed_image)
        
        # Process connected components
        filtered_contours = self._process_connected_components(
            self.preprocessed_image, 
            self.plate_img
        )
        
        # Final morphological operation
        morphological_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,3))
        final_enhanced_image = cv.dilate(filtered_contours, morphological_kernel, iterations=2)
        
        return [final_enhanced_image, self.plate_img]
