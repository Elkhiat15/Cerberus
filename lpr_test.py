
import cv2 as cv
from src.pipeline import process_image
import matplotlib.pyplot as plt

arabic_translation_map = {
    '1': '١',
    '2': '٢',
    '3': '٣',
    '4': '٤',
    '5': '٥',
    '6': '٦',
    '7': '٧',
    '8': '٨',
    '9': '٩',

    'Mem': 'م',
    'aen': 'ع',
    'alf': 'ا',
    "ba'": 'ب',
    'dal': 'د',
    "fa'": 'ف',
    'gem': 'ج',
    "ha'": 'ح',
    'lam': 'ل',
    'noon': 'ن',
    'qaf': 'ق',
    "ra'": 'ر',
    'sad': 'ص',
    'seen': 'س',
    "ta'": 'ت',
    'waw': 'و',
    "ya'": 'ي'
}

def test_license_plate(image_path):
    image = cv.imread(image_path)

    [plate_image, recognized_text] = process_image(image)

    arabic_plate = [arabic_translation_map.get(value) for value in recognized_text]
    arabic_plate.reverse()
    
    # TODO: Add try catch or figure out the cause of error
    # it produces an error if no plate is detected saying that recognized_text is a list that has no attribute '.any()'
    if recognized_text.any():
        print(f"Recognized Text: {' '.join(recognized_text)}")
        print(f"Arabic Text: {' '.join(arabic_plate)}")
        
    else:
        print("License plate text could not be recognized.")

    # Display the license plate image with contours around detected characters
    plate_image = cv.cvtColor(plate_image, cv.COLOR_BGR2RGB)  # Convert to RGB for correct display in matplotlib
    plt.imshow(plate_image)
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Path to your image file
image_path = '0028.jpg' 

# Run the test
test_license_plate(image_path)