import json
import os
import queue
import threading
import time

import cv2
import face_recognition
import numpy as np
import pyttsx3

# 🔹 Initialize the Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speech speed

speech_lock = threading.Lock()
speech_queue = queue.Queue()


# 🔹 Speech Worker Thread
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
list_names_speak = []  # Track spoken names
last_face_time = time.time()  # Track last time a face was detected
reset_timing = 10  # Reset spoken names after X seconds


# 🔹 Function to Speak Name with Control
def speak(message, name):
    with speech_lock:
        if name not in list_names_speak:  # Prevent repeated speech
            list_names_speak.append(name)
            speech_queue.put(message)  # Add name to queue for speech processing


# 🔹 Function to Reset Spoken Names
def reset_speech_list():
    global last_face_time, list_names_speak
    while True:
        time.sleep(reset_timing)  # Check every reset_timing seconds
        if time.time() - last_face_time >= reset_timing:
            list_names_speak.clear()  # Reset spoken names
            with speech_lock:
                while not speech_queue.empty():  # Clear pending speech
                    speech_queue.get()


# Start reset thread
reset_thread = threading.Thread(target=reset_speech_list, daemon=True)
reset_thread.start()

# 🔹 Load Known Faces
KNOWN_FACES_DIR = "known_faces"
AUTO_TAG_DIR = "AutoTagKnow"
os.makedirs(AUTO_TAG_DIR, exist_ok=True)  # Ensure AutoTagKnow folder exists

known_encodings = []
known_names = []


# Load welcome messages from JSON file
def load_welcome_messages(filename):
    with open(filename, 'r') as f:
        return json.load(f)


# Path to the JSON file containing welcome messages
welcome_messages_file = "welcome_messages.json"
welcome_messages = load_welcome_messages(welcome_messages_file)


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

known_encodings.extend(enc1 + enc2)
known_names.extend(names1 + names2)

# 🔹 Auto Tag Counter (Start from max existing tag)
auto_tag_counter = 1
for name in known_names:
    if name.startswith("person"):
        try:
            num = int(name[6:])  # Extract number from "person{number}"
            auto_tag_counter = max(auto_tag_counter, num + 1)
        except ValueError:
            pass  # Ignore invalid names

print("✅ Face Data Loaded. Starting Webcam...")

# 🔹 Initialize Webcam
cap = cv2.VideoCapture(0)  # Open default webcam

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# 🔹 Face Recognition Loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        last_face_time = time.time()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_encodings, face_encoding, 0.6)

        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
            name = known_names[match_index]
        else:
            # Auto-tag new face
            name = f"person{auto_tag_counter}"
            auto_tag_counter += 1

            # Save new face to AutoTagKnow folder
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(os.path.join(AUTO_TAG_DIR, f"{name}.jpg"), face_image)

            # Add to known faces
            known_encodings.append(face_encoding)
            known_names.append(name)

        if name != "Unknown":
            if name in welcome_messages:
                speak(welcome_messages[name], name)  # Speak only if not spoken recently
            else:
                speak(f"hi {name}", name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition with Auto-Tagging", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 🔹 Cleanup
cap.release()
speech_queue.put(None)  # Stop speech worker
speech_thread.join()
cv2.destroyAllWindows()
print("\n✅ Face Tracking Completed!")
