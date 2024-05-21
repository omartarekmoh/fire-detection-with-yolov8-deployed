import cv2
import requests
import numpy as np

# Flask server URL
flask_server_url = "http://127.0.0.1:8000/receive_frames/device1"  # Replace with your Flask server's IP address
cap = cv2.VideoCapture(0)


def send_frames_to_flask():
    # Use default camera (you may need to adjust the index)

    while True:
        ret, frame = cap.read()
        # if not ret:
        #     break

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
                print(response.text)
        except Exception as e:
            print("Error sending frame to Flask server:", str(e))

    # cap.release()


if __name__ == "__main__":
    send_frames_to_flask()
