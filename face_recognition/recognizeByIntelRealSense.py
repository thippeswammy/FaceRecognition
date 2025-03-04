import os
import threading

import cv2
import face_recognition
import numpy as np
import pyrealsense2 as rs

# Face Recognition Data
KNOWN_FACES_DIR = "known_faces"
known_encodings = []
known_names = []

# Load Known Faces
for filename in os.listdir(KNOWN_FACES_DIR):
    image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])

print("✅ Face Data Loaded. Starting RealSense...")

# RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # High FPS
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth

profile = pipeline.start(config)
align = rs.align(rs.stream.color)  # Align depth to color frame

# Multi-threading for Frame Capture
frames_lock = threading.Lock()
color_frame, depth_frame = None, None


def capture_frames():
    global color_frame, depth_frame
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        with frames_lock:
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()


# Start Capture Thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()


# Kalman Filter for Face Tracking
class KalmanFilter:
    def __init__(self, x, y):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)

    def predict(self):
        return self.kf.predict()[:2]

    def correct(self, x, y):
        self.kf.correct(np.array([[x], [y]], np.float32))


# Tracking Variables
prev_faces = []
prev_encodings = []

# Main Loop
while True:
    with frames_lock:
        if color_frame is None or depth_frame is None:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_data = np.asanyarray(depth_frame.get_data())

    rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Face Detection (HOG Model)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected_faces = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_encodings, face_encoding, 0.6)

        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
            name = known_names[match_index]

        # Get Depth Data for Face Center
        center_x, center_y = (left + right) // 2, (top + bottom) // 2
        if 0 <= center_x < depth_data.shape[1] and 0 <= center_y < depth_data.shape[0]:
            depth_value = depth_frame.get_distance(center_x, center_y)
        else:
            depth_value = -1  # Invalid depth

        detected_faces.append((left, top, right, bottom, name, depth_value, face_encoding))

    # Face Tracking & Kalman Filtering
    for i, (left, top, right, bottom, name, depth, encoding) in enumerate(detected_faces):
        if len(prev_faces) < len(detected_faces):
            prev_faces.append(KalmanFilter((left + right) // 2, (top + bottom) // 2))
            prev_encodings.append(encoding)

        predicted_x, predicted_y = prev_faces[i].predict().ravel()
        cv2.circle(color_image, (int(predicted_x), int(predicted_y)), 5, (255, 0, 0), -1)  # Prediction Marker
        prev_faces[i].correct((left + right) // 2, (top + bottom) // 2)

        # Draw Face Box
        cv2.rectangle(color_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(color_image, f"{name} ({depth:.2f}m)", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display
    cv2.imshow("Multi-threaded Face Recognition", color_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()

print("\n✅ Face Tracking Completed!")
