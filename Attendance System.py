import cv2
import face_recognition
import numpy as np
import os

# Load images and names of people from database
known_face_names = []
known_face_images = []

# Specify the directories where the images are located
directories = ["C:\\Users\\aarya\\OneDrive\\Desktop\\6\\Faec Recognition\\Attendance\\Elon Musk",
               "C:\\Users\\aarya\\OneDrive\\Desktop\\6\\Faec Recognition\\Attendance\\Modi"]

# Loop through all the directories
for directory in directories:
    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        # Load each image file
        img = face_recognition.load_image_file(os.path.join(directory, filename))

        # Create a face encoding for each face in the image
        face_encodings = face_recognition.face_encodings(img)

        # Loop through each face encoding
        for face_encoding in face_encodings:
            # Add the face encoding and name to the lists
            known_face_images.append(face_encoding)
            known_face_names.append(os.path.basename(directory))

# Initialize video capture device
cap = cv2.VideoCapture(0)

# Initialize list to store the names of people present
present_names = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find all the faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for i, face_encoding in enumerate(face_encodings):
        # Compare the face with the faces in the database
        matches = [face_recognition.compare_faces([known_face_encoding], face_encoding)[0] for known_face_encoding in known_face_images]

        # Find the name of the person whose face matches
        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            
            # Add the name to the list of present names if it's not already there
            if name not in present_names:
                present_names.append(name)

        # Draw a box around the face with the name of the person above the box
        top, right, bottom, left = face_locations[i]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Attendance System', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()

# Display the names of people present in the terminal
print("Names of people present:")
for name in present_names:
    print(name)
