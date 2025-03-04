import os

import cv2
import face_recognition
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable RGB and depth streams
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Load known faces
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
MODEL = "hog"  # Use 'hog' for CPU, 'cnn' for GPU

known_encodings = []
known_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, face_locations)

    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])
        print(f"Encoded: {filename}")
    else:
        print(f"Warning: No face detected in {filename}")

print("\nStarting face recognition with RealSense. Press 'q' to exit.")

while True:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # Convert RealSense frame to OpenCV format
    color_image = np.asanyarray(color_frame.get_data())

    # Detect faces in the RGB frame
    rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare detected faces with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        if True in matches:
            match_index = np.argmin(face_distances)
            name = known_names[match_index]

        # Get depth at the center of the face
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        depth_value = depth_frame.get_distance(center_x, center_y)

        # Draw rectangle around face
        cv2.rectangle(color_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(color_image, f"{name} ({depth_value:.2f}m)", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display video output
    cv2.imshow("RealSense Face Recognition", color_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
print("\n✅ Face recognition completed!")
