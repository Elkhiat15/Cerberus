import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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


def process_streaming_frame(frame, controller):
    # Process the frame with the GateAccessController
    result = controller.process_image(frame)
    
    if result.success:
        raw_plate = " ".join(result.characters)
        arabic_plate = "No Arabic characters detected"
        if result.arabic_characters:
            arabic_plate = " ".join(result.arabic_characters)
        
        print(f"Raw detected plate: {raw_plate}")
        print(f"Arabic plate: {arabic_plate}")
        
        access_granted = controller.verify_access(arabic_plate)
        print(f"Access granted: {access_granted}")
        
        cv.putText(frame, f"Plate: {raw_plate}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        
    else:
        print(f"Error: {result.error_message}")
    
    return frame, result.plate_image

def run_test_live():
    controller = GateAccessController(
        model_path="data/models/model_svm.pkl",
        authorized_plates=["1 ن ط و"],
        translation_map=arabic_translation_map,
    )

    # Open the video capture (from webcam or video file)
    IP = "http://192.168.0.127:8080/video"
    cap = cv.VideoCapture(IP)  # Use '0' for webcam or specify a video file path

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Process the frame and show the results
        frame, annotated_plate_image = process_streaming_frame(frame, controller)
        annotated_plate_image = cv.resize(annotated_plate_image, (frame.shape[1], frame.shape[0]))

        combined_image = np.vstack((frame, annotated_plate_image))

        cv.imshow("Live Stream", combined_image)
        
        # Break the loop if 'q' is pressed
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()  
    cv.destroyAllWindows()  

run_test_live()
