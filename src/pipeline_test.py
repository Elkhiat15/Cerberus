import cv2 as cv
import matplotlib.pyplot as plt
import os 
from core.pipeline import GateAccessController

arabic_translation_map = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "Mem": "م",
    "aen": "ع",
    "alf": "ا",
    "ba'": "ب",
    "dal": "د",
    "fa'": "ف",
    "gem": "ج",
    "ha'": "هـ",
    "lam": "ل",
    "noon": "ن",
    "qaf": "ق",
    "ra'": "ر",
    "sad": "ص",
    "seen": "س",
    "ta'": "ط",
    "waw": "و",
    "ya'": "ي",
}

def run_test(image_path):

    controller = GateAccessController(
        model_path="data/models/model_svm.pkl",
        authorized_plates=["1 ن ط و"],
        translation_map=arabic_translation_map,
    )

    path = os.path.join("tests", "test_images", image_path)
    # test_image = cv.imread("tests/test_images/00.jpg")
    test_image = cv.imread(path)
    result = controller.process_image(test_image)

    if result.success:
        # Get both raw and Arabic versions
        raw_plate = " ".join(result.characters)
        arabic_plate = "No Arabic characters detected"
        if result.arabic_characters:
            arabic_plate = " ".join(result.arabic_characters)

        print(f"Raw detected plate: {raw_plate}")
        print(f"Arabic plate: {arabic_plate}")

        access_granted = controller.verify_access(arabic_plate)
        print(f"Access granted: {access_granted}")

        plt.imshow(cv.cvtColor(result.plate_image, cv.COLOR_BGR2RGB))
        plt.show()

    else:
        print(f"Error: {result.error_message}")


run_test("0021.jpg")
