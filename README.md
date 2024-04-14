# Face-Recognition-Attendance-System
# Face-Recognition-Attendance-System-Using-Python
Project: Face Recognition Attendance System

Description
This Python project implements a basic face recognition system that can be used to mark attendance. The system uses OpenCV for image processing, face recognition for facial feature extraction, and NumPy for numerical computations.

Prerequisites

Python 3.x
OpenCV (pip install opencv-python)
NumPy (pip install numpy)
face_recognition (pip install face_recognition)

Setup Instructions
Clone or download the project repository.
Install required packages:

Bash
pip install opencv-python numpy face_recognition

Use code [with caution.](https://gemini.google.com/faq#coding)

Create a folder named "images" in the project directory.

Place images of individuals for whom you want to track attendance in the "images" folder. Name the images with the individual's name (e.g., "john_smith.jpg").

Usage
Run the Python script:
Bash
python attendance_system.py
Use code with caution.
Your webcam will open. Stand in front of the camera to have your face recognized and attendance marked.
Press 'q' to exit the program.


Code (attendance_system.py)
import cv2
import numpy as np
import face_recognition
import os

# Load images from the 'images' folder
path = 'images'
images = []
person_names = []
list_of_files = os.listdir(path)

for file in list_of_files:
    img = cv2.imread(os.path.join(path, file))
    images.append(img)
    person_names.append(os.path.splitext(file)[0])

# Encode the known faces
def encode_faces(images):
    encoded_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encoded_list.append(encode)
    return encoded_list

# Initialize known face encodings and names
known_face_encodings = encode_faces(images)
print("Encoding complete!")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Find faces and their encodings in the current frame 
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Mark attendance
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the closest matching face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = person_names[best_match_index]

        # Draw a box and display the name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Attendance System', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
