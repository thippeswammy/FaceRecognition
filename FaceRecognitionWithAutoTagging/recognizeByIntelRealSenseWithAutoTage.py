import json
import os
import queue
import threading
import time

import cv2
import face_recognition
import numpy as np
import pyrealsense2 as rs
import pyttsx3


def Config(filename):
    with open(filename, 'r') as f:
        return json.load(f)


# 🔹 Directories for known and auto-tagged faces
ConfigFilePath = "C:\\PythonExecution\\FaceRecognition\\ConfigFile.json"
KNOWN_FACES_DIR = "C:\\PythonExecution\\FaceRecognition\\known_faces"
AUTO_TAG_DIR = "C:\\PythonExecution\\FaceRecognition\\AutoTagKnow"

# Path to the JSON file containing welcome messages
data = Config(ConfigFilePath)
setting = data["settings"]
message = data["welcome_messages"]
tolerance = 1 - float(setting["Conf"])
reset_timing = setting["reset_timing"]

# 🔹 Initialize the Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speech speed

speech_lock = threading.Lock()
speech_queue = queue.Queue()


# 🔹 Speech Worker T hread
def speech_worker():
    while True:
        name = speech_queue.get()
        if name is None:
            break  # Stop the worker when None is received
        engine.say(name)
        engine.runAndWait()


speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# 🔹 Speech Control Variables
list_names_speak = []
last_face_time = time.time()


def speak(message, name):
    with speech_lock:
        if name not in list_names_speak:
            list_names_speak.append(name)
            speech_queue.put(message)


def reset_speech_list():
    global last_face_time, list_names_speak
    while True:
        time.sleep(reset_timing)
        if time.time() - last_face_time >= reset_timing:
            list_names_speak.clear()
            with speech_lock:
                while not speech_queue.empty():
                    speech_queue.get()


reset_thread = threading.Thread(target=reset_speech_list, daemon=True)
reset_thread.start()

os.makedirs(AUTO_TAG_DIR, exist_ok=True)  # Ensure directory exists

# 🔹 Load known faces
known_encodings = []
known_names = []
known_names_only_say = []


# Function to load faces from a given directory
def load_faces_from_directory(directory):
    encodings = []
    names = []
    for filename in os.listdir(directory):
        image = face_recognition.load_image_file(os.path.join(directory, filename))
        encodings_list = face_recognition.face_encodings(image)
        if encodings_list:
            encodings.append(encodings_list[0])
            names.append(os.path.splitext(filename)[0])
    return encodings, names


# Load known faces from both directories
enc1, names1 = load_faces_from_directory(KNOWN_FACES_DIR)
enc2, names2 = load_faces_from_directory(AUTO_TAG_DIR)
for i in names1:
    known_names_only_say.append(names1)
if setting["ENABLE_AUTO_TAG"] == "True":
    known_encodings.extend(enc1 + enc2)
    known_names.extend(names1 + names2)
else:
    known_encodings = enc1
    known_names = names1

# 🔹 Auto Tag Counter (Start from max existing tag)
auto_tag_counter = 1
for name in known_names:
    if name.startswith("person"):
        try:
            num = int(name[6:])  # Extract number from "person{number}"
            auto_tag_counter = max(auto_tag_counter, num + 1)
        except ValueError:
            pass  # Ignore invalid names
print("✅ Face Data Loaded. Starting RealSense...")

# 🔹 Initialize RealSense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

frames_lock = threading.Lock()
color_frame, depth_frame = None, None


# 🔹 Capture Frames Thread
def capture_frames():
    global color_frame, depth_frame
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        with frames_lock:
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()


capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()


# 🔹 Kalman Filter for Face Tracking
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


# 🔹 Face Recognition Loop
prev_faces = []
prev_encodings = []

while True:
    with frames_lock:
        if color_frame is None or depth_frame is None:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_data = np.asanyarray(depth_frame.get_data())

    rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        last_face_time = time.time()

    detected_faces = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance)
        center_x, center_y = (left + right) // 2, (top + bottom) // 2
        depth_value = depth_frame.get_distance(center_x, center_y) if (
                0 <= center_x < depth_data.shape[1] and 0 <= center_y < depth_data.shape[0]) else -1
        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
            name = known_names[match_index]
        else:
            if depth_value <= float(setting['DepthCon']) and setting["ENABLE_AUTO_TAG"] == "TRUE":
                name = f"person{auto_tag_counter}"
                auto_tag_counter += 1  # Increment counter for next person
                face_image = color_image[top:bottom, left:right]
                if face_image.size > 0:  # Ensure the cropped image is valid
                    save_path = os.path.join(AUTO_TAG_DIR, f"{name}.jpg")
                    cv2.imwrite(save_path, face_image)
                    print(f"✅ New person detected: {name}, saved as {save_path}")
                    known_encodings.append(face_encoding)
                    known_names.append(name)

        detected_faces.append((left, top, right, bottom, name, depth_value, face_encoding))
        if not (name == "Unknown" or name[:6] == "person"):
            if name in message:
                speak(message[name], name)
            else:
                speak(f"hi {name}", name)

    for i, (left, top, right, bottom, name, depth, encoding) in enumerate(detected_faces):
        if len(prev_faces) < len(detected_faces):
            prev_faces.append(KalmanFilter((left + right) // 2, (top + bottom) // 2))
            prev_encodings.append(encoding)

        predicted_x, predicted_y = prev_faces[i].predict().ravel()
        cv2.circle(color_image, (int(predicted_x), int(predicted_y)), 5, (255, 0, 0), -1)
        prev_faces[i].correct((left + right) // 2, (top + bottom) // 2)

        cv2.rectangle(color_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(color_image, f"{name} ({depth:.2f}m)", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    try:
        cv2.imshow("Auto-Tag Face Recognition", color_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Ended..")

# 🔹 Cleanup
pipeline.stop()
speech_queue.put(None)
speech_thread.join()
cv2.destroyAllWindows()
print("\n✅ Face Tracking Completed!")
