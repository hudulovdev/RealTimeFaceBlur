import cv2

# Load the pre-trained face cascade from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Blur the faces
    for (x, y, w, h) in faces:
        # Apply blur effect to the face region
        blurred_face = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_face

    # Display the resulting frame
    cv2.imshow('Real-Time Face Blur', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
