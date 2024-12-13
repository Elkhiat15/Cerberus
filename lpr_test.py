
import cv2 as cv
from src.pipeline import process_image
import matplotlib.pyplot as plt


def test_license_plate(image_path):
    image = cv.imread(image_path)

    [plate_image, recognized_text] = process_image(image)

    if recognized_text.any():
        print(f"Recognized Text: {' '.join(recognized_text)}")
        
    else:
        print("License plate text could not be recognized.")

    plate_image = cv.cvtColor(plate_image, cv.COLOR_BGR2RGB)  
    plt.imshow(plate_image)
    plt.axis('off')  
    plt.show()

image_path = '0028.jpg' 

test_license_plate(image_path)