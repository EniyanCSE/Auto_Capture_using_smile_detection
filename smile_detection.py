import cv2
import dlib
import numpy as np

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Flag to track if smile is detected
smile_detected = False

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray)
    
    # Iterate over detected faces
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)
        
        # Extract features for smile detection
        # Example: Calculate the distance between the corners of the mouth
        mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
        mouth_right = (landmarks.part(54).x, landmarks.part(54).y)
        mouth_width = np.abs(mouth_right[0] - mouth_left[0])
        
        # Calculate the distance between the eyes as a reference distance
        eye_left = (landmarks.part(36).x, landmarks.part(36).y)
        eye_right = (landmarks.part(45).x, landmarks.part(45).y)
        eye_distance = np.sqrt((eye_right[0] - eye_left[0])**2 + (eye_right[1] - eye_left[1])**2)
        
        # Normalize the smile width based on the eye distance
        normalized_width = (mouth_width / eye_distance) * 100  # Scale the width to a percentage
        
        # Draw a rectangle around the mouth area
        cv2.rectangle(frame, (mouth_left[0], mouth_left[1]), (mouth_right[0], mouth_right[1]), (255, 0, 0), 2)
        
        # Display the normalized width of the mouth
        cv2.putText(frame, f"Normalized Width: {normalized_width:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check if the normalized width exceeds the threshold (e.g., 75 pixels)
        if normalized_width > 70:
            smile_detected = True
    
    # If smile is detected, save the frame
    if smile_detected:
        cv2.imwrite("smile_selfie.jpg", frame)
        print("Smile detected! Selfie captured.")
        break
    
    # Display the frame
    cv2.imshow('Smile Detection', frame)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
