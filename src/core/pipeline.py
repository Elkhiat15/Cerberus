from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import load
from skimage.feature import hog

from core.character_extractor import *
from core.license_plate_detector import *
from core.plate_enhancer import *


def process_image(image) -> Tuple[np.ndarray, List[str]]:

    car_image = np.array(image)
    lpd = LicensePlateDetector()
    enhancer = LicensePlateEnhancer()
    extractor = CharExtractor()

    plate = lpd.detect(car_image)
    enhanced_plate = enhancer.enhance_plate(plate)
    car_plate, flag, car_letters = extractor.extract_chars(enhanced_plate)

    car_plate = np.array(car_plate)
    result = []
    if flag == 1:
        model = load("./data/models/model_svm.pkl", mmap_mode="r")
        data = []
        car_letters = sorted(car_letters, key=lambda x: x[1], reverse=True)
        for i in car_letters:
            car_image = cv.resize(i[0], (32, 64))
            gray = cv.cvtColor(car_image, cv.COLOR_BGR2GRAY)
            describtor = hog(
                gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1)
            )
            data.append((describtor).flatten())
        df_test = pd.DataFrame(data)
        df_test = df_test.dropna(axis=1)
        result = model.predict(df_test)
        print(result)
        flag = 0
        car_letters = []
    else:
        result = []

    return (car_plate, result)
