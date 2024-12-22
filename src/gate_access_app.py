import logging
import sys

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPalette, QPixmap
from PyQt6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QLineEdit, QListWidget, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

from core.pipeline import GateAccessController

logger = logging.getLogger(__name__)


class DarkPalette(QPalette):
    def __init__(self):
        super().__init__()
        self.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        self.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)


class GateAccessApp(QMainWindow):
    def __init__(self):
        super().__init__()
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

        self.controller = GateAccessController(translation_map=arabic_translation_map)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Gate Access Control")
        self.setGeometry(100, 100, 1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)

        input_frame = QFrame()
        input_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        input_layout = QVBoxLayout(input_frame)

        title_label = QLabel("Plate Management")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.plate_input = QLineEdit()
        self.plate_input.setPlaceholderText("Enter plate number")
        self.plate_input.setStyleSheet(
            """
            QLineEdit {
                padding: 8px;
                border: 2px solid #444;
                border-radius: 4px;
                background-color: #2a2a2a;
            }
        """
        )

        button_layout = QHBoxLayout()
        add_button = QPushButton("Add Plate")
        remove_button = QPushButton("Remove Plate")

        for button in [add_button, remove_button]:
            button.setStyleSheet(
                """
                QPushButton {
                    padding: 8px 15px;
                    background-color: #2a2a2a;
                    border: 2px solid #444;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #3a3a3a;
                }
                QPushButton:pressed {
                    background-color: #222;
                }
            """
            )

        add_button.clicked.connect(self.add_plate)
        remove_button.clicked.connect(self.remove_plate)
        button_layout.addWidget(add_button)
        button_layout.addWidget(remove_button)

        input_layout.addWidget(title_label)
        input_layout.addWidget(self.plate_input)
        input_layout.addLayout(button_layout)

        # Authorized plates list
        list_label = QLabel("Authorized Plates")
        list_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        list_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.plates_list = QListWidget()
        self.plates_list.setStyleSheet(
            """
            QListWidget {
                background-color: #2a2a2a;
                border: 2px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #3a3a3a;
            }
        """
        )

        left_layout.addWidget(input_frame)
        left_layout.addWidget(list_label)
        left_layout.addWidget(self.plates_list)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet(
            """
            QLabel {
                border: 2px solid #444;
                border-radius: 4px;
                background-color: #2a2a2a;
            }
        """
        )

        # Image control buttons
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load Image")
        process_button = QPushButton("Process Image")

        for button in [load_button, process_button]:
            button.setStyleSheet(
                """
                QPushButton {
                    padding: 10px 20px;
                    background-color: #2a2a2a;
                    border: 2px solid #444;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3a3a3a;
                }
                QPushButton:pressed {
                    background-color: #222;
                }
            """
            )

        load_button.clicked.connect(self.load_image)
        process_button.clicked.connect(self.process_image)
        button_layout.addWidget(load_button)
        button_layout.addWidget(process_button)

        # Results display
        self.result_label = QLabel("Results will appear here")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet(
            """
            QLabel {
                padding: 10px;
                background-color: #2a2a2a;
                border: 2px solid #444;
                border-radius: 4px;
            }
        """
        )
        self.result_label.setMinimumHeight(100)

        self.result_image = QLabel()
        self.result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image.setMinimumSize(600, 400)
        self.result_image.setStyleSheet(
            """
            QLabel {
                border: 2px solid #444;
                border-radius: 4px;
                background-color: #2a2a2a;
            }
        """
        )

        right_layout.addWidget(self.image_label)
        right_layout.addLayout(button_layout)
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.result_image)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        self.current_image = None
        self.update_plates_list()

    def update_plates_list(self):
        self.plates_list.clear()
        for plate in sorted(self.controller.authorized_plates):
            self.plates_list.addItem(plate[::-1])

    def add_plate(self):
        plate = self.plate_input.text().strip()
        plate = plate[::-1]
        logger.info(f"Adding plate: {plate}")
        print(f"Adding plate: {plate}")

        if plate:
            self.controller.add_authorized_plate(plate)
            QMessageBox.information(
                self, "Success", f"Plate {plate} added to authorized list"
            )
            self.plate_input.clear()
            self.update_plates_list()
        else:
            QMessageBox.warning(self, "Error", "Please enter a plate number")

    def remove_plate(self):
        plate = self.plate_input.text().strip()
        if plate:
            self.controller.remove_authorized_plate(plate[::-1])
            QMessageBox.information(
                self, "Success", f"Plate {plate} removed from authorized list"
            )
            self.plate_input.clear()
            self.update_plates_list()
        else:
            QMessageBox.warning(self, "Error", "Please enter a plate number")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp)"
        )

        if file_name:
            self.current_image = cv2.imread(file_name)
            if self.current_image is not None:
                rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w

                qt_image = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

                self.image_label.setPixmap(scaled_pixmap)
                self.result_label.setText(
                    'Image loaded. Click "Process Image" to analyze.'
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")

    def process_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "Please load an image first")
            return

        result = self.controller.process_image(self.current_image)

        if result.success:
            plate_text = " ".join(result.arabic_characters)
            is_authorized = self.controller.verify_access(plate_text)

            status = "AUTHORIZED" if is_authorized else "UNAUTHORIZED"
            status_color = "#4CAF50" if is_authorized else "#F44336"  # Green or Red

            result_text = f"""
            <div style='font-size: 14px;'>
                <p><b>Detected Plate:</b> {result.characters}</p>
                <p><b>Status:</b> <span style='color: {status_color};'>{status}</span></p>
            """
            if result.arabic_characters:
                arabic_text = " ".join(result.arabic_characters)
                result_text += f"<p><b>Arabic:</b> {arabic_text[::-1]}</p>"

            result_text += "</div>"

            self.result_label.setText(result_text)


            if result.plate_image is not None:
                rgb_image = cv2.cvtColor(result.plate_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w

                qt_image = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.result_image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.result_image.setPixmap(scaled_pixmap)
        else:
            self.result_label.setText(
                f"<p style='color: #F44336;'>Error: {result.error_message}</p>"
            )


def main():
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    palette = DarkPalette()
    app.setPalette(palette)

    app.setStyleSheet(
        """
        QMainWindow {
            background-color: #333;
        }
        QLabel {
            color: white;
        }
        QMessageBox {
            background-color: #333;
        }
        QMessageBox QLabel {
            color: white;
        }
    """
    )

    window = GateAccessApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
