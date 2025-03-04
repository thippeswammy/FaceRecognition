import os
import queue
import threading
import time

import cv2
import face_recognition
import numpy as np
import pyttsx3

# Initialize the speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speech speed

# Directory containing images of known people
KNOWN_FACES_DIR = "known_faces"

conf = 0.6
# Tolerance for matching (Lower = more strict, Higher = more lenient)
TOLERANCE = 1 - conf

reset_timing = 5
# Use "hog" for CPU, "cnn" for GPU (hog is slower but works on all machines)
MODEL = "hog"

# Load and encode known faces
known_encodings = []
known_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)

    # Detect face locations before encoding
    face_locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, face_locations)

    if encodings:
        known_encodings.append(encodings[0])  # Store the first detected face encoding
        known_names.append(os.path.splitext(filename)[0])  # Store the person's name
        print(f"Encoded: {filename}")
    else:
        print(f"Warning: No face detected in {filename}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam

print("\nStarting live face recognition. Press 'q' to exit.")
last_spoken_name = None  # To prevent repeated speech
list_names_speak = []
last_face_time = time.time()  # Track the last time a face was detected
speech_lock = threading.Lock()  # Lock for speech synchronization
speech_queue = queue.Queue()  # Queue for speech handling


# Function to process speech from the queue
def speech_worker():
    while True:
        name = speech_queue.get()
        if name is None:
            break  # Stop the worker when None is received
        engine.say(name)
        engine.runAndWait()


# Start a background thread for speech synthesis
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


# Function to add names to the speech queue
def speak(name):
    with speech_lock:
        if name not in list_names_speak:
            list_names_speak.append(name)
            speech_queue.put(name)  # Add to queue instead of running directly


# Function to check if no face is detected for 2 seconds
def check_no_face():
    global last_face_time, list_names_speak
    while True:
        time.sleep(reset_timing)  # Check every 2 seconds
        if time.time() - last_face_time >= reset_timing:
            list_names_speak.clear()  # Clear spoken names
            with speech_lock:
                while not speech_queue.empty():  # Clear speech queue properly
                    speech_queue.get()


# Start the no-face detection thread
threading.Thread(target=check_no_face, daemon=True).start()

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to RGB (face_recognition uses RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encode them
    face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:  # If faces are found, update last face detection time
        last_face_time = time.time()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"

        if True in matches:
            match_index = np.argmin(face_distances)  # Get the best match
            name = known_names[match_index]

        # Speak the detected name only if it changes
        # if name != "Unknown":
        speak(name)  # Add name to speech queue
        last_spoken_name = name  # Update last spoken name

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Live Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
speech_queue.put(None)  # Stop speech worker
speech_thread.join()
video_capture.release()
cv2.destroyAllWindows()
