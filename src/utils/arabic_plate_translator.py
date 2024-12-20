from typing import Dict, List


class ArabicPlateTranslator:
    """Handles translation between model predictions and Arabic characters"""

    def __init__(self, translation_map: Dict[str, str]):
        self.translation_map = translation_map
        self.reverse_map = {v: k for k, v in translation_map.items()}

    def translate_to_arabic(self, predictions: List[str]) -> List[str]:
        return [self.translation_map.get(pred, pred) for pred in predictions]

    def translate_to_model_classes(self, arabic_chars: List[str]) -> List[str]:
        return [self.reverse_map.get(char, char) for char in arabic_chars]
