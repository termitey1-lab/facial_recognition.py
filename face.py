import cv2
from deepface import DeepFace
import numpy as np


def recognize_face_from_camera():
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit the camera stream.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Detect and analyze face attributes using DeepFace
        try:
            result = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False
            )

            # Extract analysis
            dominant_emotion = result[0]['dominant_emotion']
            age = result[0]['age']
            gender = result[0]['dominant_gender']

            # Overlay detected info on the frame
            info = f"Age: {age} | Gender: {gender} | Emotion: {dominant_emotion}"
            cv2.putText(frame, info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            # Skip frames where no faces are detected
            print(f"No face detected: {e}")

        # Display the resulting frame
        cv2.imshow('Facial Recognition (DeepFace + OpenCV)', frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_face_from_camera()

