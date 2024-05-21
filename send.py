import cv2
import requests
import numpy as np

# Flask server URL
flask_server_url = "http://127.0.0.1:8000/receive_frames/device1"  # Replace with your Flask server's IP address
cap = cv2.VideoCapture(0)  # Initialize video capture object for the default camera

def send_frames_to_flask():
    """
    Capture frames from the webcam and send them to the Flask server.
    """
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Failed to capture frame from webcam")
            break

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        # Send frame to Flask server
        try:
            response = requests.post(
                flask_server_url,
                data=frame_bytes,
                headers={"Content-Type": "image/jpeg"},
            )
            if response.status_code != 200:
                print("Failed to send frame to Flask server:", response.status_code)
            else:
                print(response.text)  # Print the response text from the server
        except Exception as e:
            print("Error sending frame to Flask server:", str(e))

if __name__ == "__main__":
    send_frames_to_flask()  # Start capturing and sending frames to the Flask server
