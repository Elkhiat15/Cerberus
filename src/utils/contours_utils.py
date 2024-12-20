import cv2 as cv
import numpy as np


def calculate_contour_metrics(contour1, contour2, mode):
    """
    Calculate distance metrics between two contours with flexible modes

    Modes:
    - 'x': Horizontal clustering
    - 'y': Vertical clustering
    """
    rect1_x, rect1_y, rect1_width, rect1_height = cv.boundingRect(contour1)
    rect2_x, rect2_y, rect2_width, rect2_height = cv.boundingRect(contour2)

    if mode == "x":
        center1_x = rect1_x + rect1_width
        center1_y = rect1_y + rect1_height / 2
        center2_x = rect2_x
        center2_y = rect2_y + rect2_height / 2

        if center1_x > center2_x:
            center1_x = rect1_x
            center2_x = rect2_x + rect2_width

    elif mode == "y":
        center1_x = rect1_x + rect1_width / 2
        center1_y = rect1_y + rect1_height
        center2_x = rect2_x + rect2_width / 2
        center2_y = rect2_y

        if center1_y > center2_y:
            center1_y = rect1_y
            center2_y = rect2_y + rect2_height

    return (abs(center1_x - center2_x), abs(center1_y - center2_y))


def agglomerative_cluster(contours, mode, threshold_distance=20, y_threshold=25):
    """
    Generic agglomerative clustering with flexible mode

    Modes:
    - 'x': Horizontal clustering
    - 'y': Vertical clustering
    """
    clustered_contours = contours

    while len(clustered_contours) > 1:
        min_distance = None
        closest_contour_indices = None

        for x in range(len(clustered_contours) - 1):
            for y in range(x + 1, len(clustered_contours)):
                horizontal_distance, vertical_distance = calculate_contour_metrics(
                    clustered_contours[x], clustered_contours[y], mode=mode
                )

                horizontal_condition = (
                    horizontal_distance <= 10 if mode == "y" else True
                )
                vertical_condition = (
                    vertical_distance <= y_threshold if mode == "x" else True
                )

                if (min_distance is None and vertical_condition) or (
                    min_distance is not None
                    and (horizontal_distance if mode == "x" else vertical_distance)
                    < min_distance
                    and vertical_condition
                    and horizontal_condition
                ):
                    min_distance = (
                        horizontal_distance if mode == "x" else vertical_distance
                    )
                    closest_contour_indices = (x, y)

        if min_distance is not None and min_distance < threshold_distance:
            index1, index2 = closest_contour_indices
            clustered_contours[index1] = np.concatenate(
                (clustered_contours[index1], clustered_contours[index2]), axis=0
            )
            del clustered_contours[index2]
        else:
            break

    return clustered_contours


# Function to check if two contours intersect
def check_contours_intersection(cnt1, cnt2, x_threshold=0):
    rect1 = cv.boundingRect(cnt1)
    rect2 = cv.boundingRect(cnt2)
    rect1_x, rect1_y, rect1_width, rect1_height = rect1
    rect2_x, rect2_y, rect2_width, rect2_height = rect2

    # Check if there is no intersection in the x-direction with the given threshold
    if (
        rect1_x + rect1_width + x_threshold < rect2_x
        or rect2_x + rect2_width + x_threshold < rect1_x
    ):
        return False

    # Check for intersection in the y-direction
    return not (rect1_y + rect1_height < rect2_y or rect2_y + rect2_height < rect1_y)


# Function to merge intersecting contours
def merge_intersecting_contours(contours):
    merged_contours = []
    processed_contours = set()

    for i, current_contour in enumerate(contours):
        if i not in processed_contours:
            merged = current_contour.copy()

            for j, comparison_contour in enumerate(contours):
                if (
                    i != j
                    and j not in processed_contours
                    and check_contours_intersection(current_contour, comparison_contour)
                ):
                    merged = np.concatenate((merged, comparison_contour))
                    processed_contours.add(j)

            merged_contours.append(merged)

    return merged_contours
