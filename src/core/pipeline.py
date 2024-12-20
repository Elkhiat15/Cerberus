import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from joblib import load
from skimage.feature import hog

from core.character_extractor import CharExtractor
from core.license_plate_detector import LicensePlateDetector
from core.plate_enhancer import LicensePlateEnhancer
from utils.arabic_plate_translator import ArabicPlateTranslator


@dataclass
class PlateRecognitionResult:

    plate_image: np.ndarray
    success: bool
    characters: List[str]
    arabic_characters: Optional[List[str]] = None
    error_message: Optional[str] = None


class GateAccessController:

    def __init__(
        self,
        model_path: str = "./data/models/model_svm.pkl",
        authorized_plates: List[str] = [],
        translation_map: dict = {},
    ):
        self.logger = self._setup_logger()
        self.model_path = Path(model_path)
        self.detector = LicensePlateDetector()
        self.enhancer = LicensePlateEnhancer()
        self.extractor = CharExtractor()
        self.translator = ArabicPlateTranslator(translation_map)
        self.model = self._load_model()
        self.authorized_plates = set(authorized_plates)

        # HOG parameters
        self.hog_params = {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (1, 1),
        }

        # Character dimensions
        self.char_dims = (32, 64)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("GateAccessControl")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _load_model(self) -> object:
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            return load(self.model_path, mmap_mode="r")
        except FileNotFoundError as e:
            self.logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _extract_hog_features(self, char_image: np.ndarray) -> np.ndarray:
        try:
            resized = cv2.resize(char_image, self.char_dims)

            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized

            features = hog(gray, **self.hog_params)

            return features.flatten()

        except Exception as e:
            self.logger.error(f"Error extracting HOG features: {e}")
            raise

    def process_image(self, image: np.ndarray) -> PlateRecognitionResult:
        try:
            self.logger.info("Starting image processing")

            car_image = np.array(image)

            plate = self.detector.detect(car_image)
            if plate is None:
                return PlateRecognitionResult(
                    plate_image=car_image,
                    characters=[],
                    success=False,
                    error_message="No license plate detected",
                )

            enhanced_plate = self.enhancer.enhance_plate(plate)

            car_plate, success_flag, char_regions = self.extractor.extract_chars(
                enhanced_plate
            )

            if not success_flag:
                return PlateRecognitionResult(
                    plate_image=car_plate,
                    characters=[],
                    success=False,
                    error_message="Character extraction failed",
                )

            char_features = []
            char_regions = sorted(char_regions, key=lambda x: x[1])

            for char_img, _ in char_regions:
                features = self._extract_hog_features(char_img)
                char_features.append(features)

            df_features = pd.DataFrame(char_features)
            df_features = df_features.dropna(axis=1)

            predicted_chars = self.model.predict(df_features)

            arabic_chars = self.translator.translate_to_arabic(predicted_chars)

            self.logger.info(
                f"Successfully processed image. Found {len(predicted_chars)} characters"
            )

            return PlateRecognitionResult(
                plate_image=car_plate,
                characters=list(predicted_chars),
                arabic_characters=arabic_chars,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return PlateRecognitionResult(
                plate_image=image,
                characters=[],
                success=False,
                error_message=str(e),
            )

    def verify_access(self, plate_number: str) -> bool:
        return plate_number in self.authorized_plates

    def add_authorized_plate(self, plate_number: str) -> None:
        self.authorized_plates.add(plate_number)

    def remove_authorized_plate(self, plate_number: str) -> None:
        self.authorized_plates.discard(plate_number)
