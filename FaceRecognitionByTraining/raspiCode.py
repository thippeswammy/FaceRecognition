import threading
import time

import RPi.GPIO as GPIO
import cv2
from ultralytics import YOLO

# Setup GPIO pins
LED_PIN1 = 17  # GPIO pin for LED 1
LED_PIN2 = 27  # GPIO pin for LED 2
LED_PIN3 = 22  # GPIO pin for LED 3
LED_PIN4 = 23  # GPIO pin for LED 4

GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
GPIO.setup(LED_PIN1, GPIO.OUT)
GPIO.setup(LED_PIN2, GPIO.OUT)
GPIO.setup(LED_PIN3, GPIO.OUT)
GPIO.setup(LED_PIN4, GPIO.OUT)


def set_led_state(state):
    """Controls LEDs based on the received state."""
    if state == 1:  # Blinking LEDs
        for _ in range(10):  # Blink 10 times
            GPIO.output(LED_PIN1, GPIO.HIGH)
            GPIO.output(LED_PIN2, GPIO.HIGH)
            GPIO.output(LED_PIN3, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(LED_PIN1, GPIO.LOW)
            GPIO.output(LED_PIN2, GPIO.LOW)
            GPIO.output(LED_PIN3, GPIO.LOW)
            time.sleep(0.2)
    elif state == 2:  # Detection confirmed
        GPIO.output(LED_PIN4, GPIO.HIGH)
        time.sleep(5)
        GPIO.output(LED_PIN4, GPIO.LOW)
    else:  # Default state
        GPIO.output(LED_PIN1, GPIO.LOW)
        GPIO.output(LED_PIN2, GPIO.LOW)
        GPIO.output(LED_PIN3, GPIO.LOW)
        GPIO.output(LED_PIN4, GPIO.LOW)


def get_name(result, threshold=0.5):
    """Returns the classified name and confidence."""
    names = result.names
    top1_index = result.probs.top1
    top1_prob = result.probs.top1conf.item()
    if top1_prob > threshold:
        return names[top1_index], top1_prob * 100
    else:
        return "Unknown", top1_prob * 100


def FaceRecognition(model_path):
    """Main function for face recognition and LED control."""
    face_detector = YOLO('../yolov8m-face.pt')  # Face detection model
    face_classifier = YOLO(model_path)  # Face classification model

    camera = cv2.VideoCapture(0)  # Open the default camera
    camera.set(3, 640)  # Set video width
    camera.set(4, 480)  # Set video height

    tracked_faces = {}  # Dictionary to track faces across frames
    face_id_counter = 0  # Unique ID generator for new faces

    try:
        while True:
            ret, img = camera.read()
            if not ret:
                break

            # Perform face detection
            faces_results = face_detector.predict(img, show=False, save=False, conf=0.75, stream=True)
            current_frame_faces = []

            for face_result in faces_results:
                faces = face_result.boxes.xyxy
                for (x1, y1, x2, y2) in faces:
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    detect_face = img[y1:y2, x1:x2].astype('uint8')
                    # Perform face classification
                    results_classifier = face_classifier.predict(detect_face, show=False, save=False, conf=0.7,
                                                                 stream=True)

                    for result_classifier in results_classifier:
                        name, confidence = get_name(result_classifier)
                        cv2.putText(img, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    # Assign a unique ID or match with an existing ID
                    matched_id = None
                    for face_id, face_data in tracked_faces.items():
                        prev_x1, prev_y1, prev_x2, prev_y2, count = face_data
                        if abs(x1 - prev_x1) < 50 and abs(y1 - prev_y1) < 50:  # Threshold for matching
                            matched_id = face_id
                            break
                    if matched_id is None:
                        matched_id = face_id_counter
                        face_id_counter += 1
                        tracked_faces[matched_id] = [x1, y1, x2, y2, 0]
                    tracked_faces[matched_id][:4] = [x1, y1, x2, y2]  # Update coordinates
                    tracked_faces[matched_id][4] += 1  # Increment frame count

                    # Trigger actions based on face tracking
                    if tracked_faces[matched_id][4] >= 50:
                        thread = threading.Thread(target=set_led_state, args=(1,))
                        thread.start()
                        time.sleep(5)
                        thread = threading.Thread(target=set_led_state, args=(0,))
                        thread.start()
                        del tracked_faces[matched_id]
                    elif tracked_faces[matched_id][4] > 50:
                        thread = threading.Thread(target=set_led_state, args=(2,))
                        thread.start()
                        time.sleep(5)
                        thread = threading.Thread(target=set_led_state, args=(0,))
                        thread.start()
                        del tracked_faces[matched_id]
                    current_frame_faces.append(matched_id)

            # Remove faces not seen in the current frame
            for face_id in list(tracked_faces.keys()):
                if face_id not in current_frame_faces:
                    del tracked_faces[face_id]

            cv2.imshow("Face Recognition", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()  # Cleanup GPIO


# Run the face recognition function
FaceRecognition("path_to_your_face_classifier.pt")
