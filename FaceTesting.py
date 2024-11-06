import cv2
from ultralytics import YOLO


def FaceRecognition(model_path, camera, faster=False):
    face_detector = YOLO('yolov8m-face.pt')  # Assuming this initializes your face detection model
    face_classifier = YOLO(model_path)  # Assuming this initializes your face classification model

    if not faster:
        camera = cv2.VideoCapture(0)
        camera.set(3, 640)  # set video width
        camera.set(4, 480)  # set video height
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'XVID' or 'MJPG'
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(camera.get(3)), int(camera.get(4))))

    while True:
        ret, img = camera.read()
        if not ret:
            break

        # Perform face detection
        faces_results = face_detector.predict(img, show=False, save=False, conf=0.7, stream=True)

        for face_result in faces_results:
            faces = face_result.boxes.xyxy
            for (x1, y1, x2, y2) in faces:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                detect_face = img[y1:y2, x1:x2].astype('uint8')
                # Perform face classification on detected face
                results_classifier = face_classifier.predict(detect_face, show=False, save=False, conf=0.7, stream=True)
                for result_classifier in results_classifier:
                    name, confidence = get_name(result_classifier)
                    cv2.putText(img, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)
        cv2.imshow("Face Recognition", img)
        out.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
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
