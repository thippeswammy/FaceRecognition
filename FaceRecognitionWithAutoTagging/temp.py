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
ConfigFilePath = "ConfigFile.json"
KNOWN_FACES_DIR = "known_faces"
AUTO_TAG_DIR = "AutoTagKnow"

data = Config(ConfigFilePath)
setting = data["settings"]
message = data["welcome_messages"]
tolerance = 1 - float(setting["Conf"])
reset_timing = setting["reset_timing"]

# 🔹 Initialize Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

speech_lock = threading.Lock()
speech_queue = queue.Queue()


# 🔹 Speech Worker Thread
def speech_worker():
    while True:
        name = speech_queue.get()
        if name is None:
            break
        engine.say(name)
        engine.runAndWait()


speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


def speak(message, name):
    with speech_lock:
        speech_queue.put(message)


def reset_speech_list():
    while True:
        time.sleep(reset_timing)
        with speech_lock:
            while not speech_queue.empty():
                speech_queue.get()


reset_thread = threading.Thread(target=reset_speech_list, daemon=True)
reset_thread.start()

os.makedirs(AUTO_TAG_DIR, exist_ok=True)


# 🔹 Load known faces
def load_faces(directory):
    encodings, names = [], []
    for filename in os.listdir(directory):
        image = face_recognition.load_image_file(os.path.join(directory, filename))
        enc_list = face_recognition.face_encodings(image)
        if enc_list:
            encodings.append(enc_list[0])
            names.append(os.path.splitext(filename)[0])
    return encodings, names


enc1, names1 = load_faces(KNOWN_FACES_DIR)
enc2, names2 = load_faces(AUTO_TAG_DIR)

known_encodings = enc1 + enc2 if setting["ENABLE_AUTO_TAG"] == "True" else enc1
known_names = names1 + names2 if setting["ENABLE_AUTO_TAG"] == "True" else names1

# 🔹 Auto Tag Counter
auto_tag_counter = max([int(n[6:]) for n in known_names if n.startswith("person")] + [1])

print("✅ Face Data Loaded. Starting RealSense...")

# 🔹 Initialize RealSense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

frame_queue = queue.Queue(maxsize=5)


# 🔹 Frame Capture Thread
def capture_frames():
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if color_frame:
            frame_queue.put(np.asanyarray(color_frame.get_data()))


capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()


# 🔹 Face Recognition Worker
def process_faces():
    global auto_tag_counter
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance)
            if True in matches:
                name = known_names[np.argmin(face_recognition.face_distance(known_encodings, face_encoding))]
            elif setting["ENABLE_AUTO_TAG"] == "True":
                name = f"person{auto_tag_counter}"
                auto_tag_counter += 1
                cv2.imwrite(os.path.join(AUTO_TAG_DIR, f"{name}.jpg"), frame[top:bottom, left:right])
                known_encodings.append(face_encoding)
                known_names.append(name)

            if name in message:
                speak(message[name], name)
            else:
                speak(f"Hi {name}", name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Parallel Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


process_thread = threading.Thread(target=process_faces, daemon=True)
process_thread.start()

capture_thread.join()
frame_queue.put(None)
process_thread.join()

pipeline.stop()
speech_queue.put(None)
speech_thread.join()
cv2.destroyAllWindows()
print("\n✅ Face Tracking Completed!")
