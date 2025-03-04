import threading
import time

import cv2
from ultralytics import YOLO

doCommunication = False
if doCommunication:
    from FaceRecognitionByTraining import communication as com


def FaceRecognition(model_path, camera, faster=False):
    face_detector = YOLO('../yolov8m-face.pt')  # Face detection model
    face_classifier = YOLO(model_path)  # Face classification model

    if not faster:
        camera = cv2.VideoCapture(0)
        camera.set(3, 640)  # Set video width
        camera.set(4, 480)  # Set video height
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(camera.get(3)), int(camera.get(4))))

    tracked_faces = {}  # Dictionary to track faces across frames
    face_id_counter = 0  # Unique ID generator for new faces
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
                results_classifier = face_classifier.predict(detect_face, show=False, save=False, conf=0.7, stream=True)
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

                if doCommunication and tracked_faces[matched_id][4] >= 50:
                    thread = threading.Thread(target=com.sendCommend, args=(1,))
                    thread.start()
                    time.sleep(5)
                    thread = threading.Thread(target=com.sendCommend, args=(0,))
                    thread.start()
                    del tracked_faces[matched_id]
                elif doCommunication and tracked_faces[matched_id][4] > 50:
                    thread = threading.Thread(target=com.sendCommend, args=(2,))
                    thread.start()
                    time.sleep(5)
                    thread = threading.Thread(target=com.sendCommend, args=(0,))
                    thread.start()
                    del tracked_faces[matched_id]
                current_frame_faces.append(matched_id)
        # Remove faces not seen in the current frame
        for face_id in list(tracked_faces.keys()):
            if face_id not in current_frame_faces:
                del tracked_faces[face_id]

        cv2.imshow("Face Recognition", img)
        out.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    out.release()
    cv2.destroyAllWindows()


def get_name(result, threshold=0.5):
    # Set a confidence threshold for known faces
    names = result.names
    top1_index = result.probs.top1
    top1_prob = result.probs.top1conf.item()
    if top1_prob > threshold:
        return names[top1_index], top1_prob * 100
    else:
        return "Unknown", top1_prob * 100
