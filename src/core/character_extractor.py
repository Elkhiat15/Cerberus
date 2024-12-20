import cv2 as cv
import imutils
import numpy as np
from skimage import measure

from utils.contours_utils import *


class CharExtractor:
    def __init__(self):
        self.filtered_contours = []
        self.extracted_characters = []
        self.annotated_image = None

    def extract_chars(self, img):
        self.filtered_contours = []
        self.extracted_characters = []
        self.annotated_image = None

        if img == []:
            return img

        self.img = img
        self.annotated_image = img[1].copy()

        # Preprocess the image
        self.img[0] = self.preprocess_image(self.img[0])

        # Find contours and filter them
        labels = measure.label(self.img[0], background=0)
        self.filtered_contours = self.find_and_filter_contours(labels)

        # Merge and cluster contours
        self.filtered_contours = self.merge_and_cluster_contours()

        # Extract car letters
        self.extracted_characters = self.extract_car_letters()

        # Determine is_valid_plate based on the number of car letters
        is_valid_plate = self.determine_is_valid_plate()

        return [self.img[1], is_valid_plate, self.extracted_characters]

    def preprocess_image(self, image):
        dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
        return cv.dilate(image, dilation_kernel, iterations=1)

    def find_and_filter_contours(self, labels):
        for idx, label in enumerate(np.unique(labels)):
            if label == 0:
                continue

            current_label_mask = np.zeros(self.img[0].shape, dtype="uint8")
            current_label_mask[labels == label] = 255

            label_contours = cv.findContours(
                current_label_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            label_contours = (
                label_contours[1] if imutils.is_cv3() else label_contours[0]
            )

            if len(label_contours) > 0:
                c = max(label_contours, key=cv.contourArea)
                if self.is_valid_contour(c):
                    self.filtered_contours.append(c)

        return self.filtered_contours

    def is_valid_contour(self, c):
        (contour_x, contour_y, contour_width, contour_height) = cv.boundingRect(c)
        aspect_ratio = contour_width / float(contour_height)
        contour_solidity = cv.contourArea(c) / float(contour_width * contour_height)
        area = cv.contourArea(c)

        is_valid_area = area > 70 and area < 1100
        center_line = self.img[1].shape[0] // 2

        cv.rectangle(
            self.annotated_image,
            (contour_x, contour_y),
            (contour_x + contour_width, contour_y + contour_height),
            (0, 255, 0),
            1,
        )

        return (
            is_valid_area
            and contour_solidity < 0.8
            and (
                (center_line > contour_y and center_line < contour_y + contour_height)
                or (center_line <= contour_y)
            )
        )

    def merge_and_cluster_contours(self):
        self.filtered_contours = agglomerative_cluster(self.filtered_contours, "y")
        self.filtered_contours = merge_intersecting_contours(self.filtered_contours)
        return agglomerative_cluster(self.filtered_contours, "x", threshold_distance=2)

    def extract_car_letters(self):
        for cnt in self.filtered_contours:
            (char_x, char_y, char_width, char_height) = cv.boundingRect(cnt)
            character_image = self.img[1].copy()[
                char_y : char_y + char_height, char_x : char_x + char_width
            ]
            self.extracted_characters.append((character_image, char_x))
            cv.rectangle(
                self.img[1],
                (char_x, char_y),
                (char_x + char_width, char_y + char_height),
                (0, 255, 0),
                1,
            )

        return self.extracted_characters

    def determine_is_valid_plate(self):
        if 2 <= len(self.extracted_characters) <= 7:
            return 1
        else:
            print(len(self.extracted_characters))
            return 0
