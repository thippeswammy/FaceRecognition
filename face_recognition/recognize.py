import os

import cv2
import face_recognition
import numpy as np

# Directory containing images of known people
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_IMAGE_PATH = "person.png"

# Tolerance for matching (Lower = more strict, Higher = more lenient)
TOLERANCE = 0.6

# Use "hog" for CPU, "cnn" for GPU (hog is faster, cnn is more accurate)
MODEL = "hog"

# Lists to store known face encodings and names
known_encodings = []
known_names = []

# Load and encode known faces
print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)

    # Detect face locations before encoding
    face_locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, face_locations)

    if encodings:
        known_encodings.append(encodings[0])  # Store the first detected face encoding
        known_names.append(os.path.splitext(filename)[0])  # Store the person's name (filename without extension)
        print(f"Encoded: {filename}")
    else:
        print(f"Warning: No face detected in {filename}")

# Load unknown image for recognition
print("\nLoading unknown image...")
unknown_image = face_recognition.load_image_file(UNKNOWN_IMAGE_PATH)
unknown_face_locations = face_recognition.face_locations(unknown_image, model=MODEL)
unknown_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)

if not unknown_encodings:
    print("Error: No face detected in the unknown image.")
else:
    unknown_encoding = unknown_encodings[0]

    # Compare unknown face with known faces
    print("\nComparing with known faces...")
    results = face_recognition.compare_faces(known_encodings, unknown_encoding, TOLERANCE)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

    if True in results:
        match_index = np.argmin(face_distances)  # Get the best match
        recognized_person = known_names[match_index]
        print(f"Person recognized: {recognized_person}")
    else:
        print("No match found!")

# Display the unknown image with face bounding box
for (top, right, bottom, left) in unknown_face_locations:
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow("Unknown Image", cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
