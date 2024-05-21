from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from process import _model

app = Flask(__name__)

# Dictionary to store frames from different devices
camera_frames = {}

# Load the YOLO model and move it to GPU (if available)
model = YOLO("yolo/best.pt")
model.to("cuda")

def generate_frames():
    """
    Generator function to yield frames in byte format for streaming.
    Iterates through the stored frames in `camera_frames` and encodes them as JPEG.
    """
    while True:
        for device_id, frame in camera_frames.items():
            # Resize frame to a smaller resolution (e.g., 640x480)
            # frame, fire, x, y = model2(model, frame)  # Process frame with model (commented out)

            # Encode frame as JPEG with compression quality (0-100)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Adjust quality as needed
            _, buffer = cv2.imencode(".jpg", frame, encode_param)
            frame_bytes = buffer.tobytes()

            # Yield the frame in byte format
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

@app.route("/")
def index():
    """
    Route to render the main HTML page.
    """
    return render_template("new.html")

@app.route("/video_feed")
def video_feed():
    """
    Route to stream video feed.
    """
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/receive_frames/<device_id>", methods=["POST"])
def receive_frames(device_id):
    """
    Route to receive frames from devices.
    Converts the received frame data to an image, processes it with YOLO, and stores it.
    
    Args:
        device_id (str): Identifier for the device sending frames.
    
    Returns:
        JSON response with fire detection results and coordinates.
    """
    frame_data = request.data

    # Convert received data to numpy array
    nparr = np.frombuffer(frame_data, np.uint8)

    # Decode numpy array as image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 480))
    
    # Process the image with YOLO model
    result = model(img)
    frame, fire, x, y = _model(result, img)

    # Store the processed frame for the corresponding device
    camera_frames[device_id] = frame

    return jsonify({"fire": fire, "x": x, "y": y})

if __name__ == "__main__":
    # Start the Flask application
    app.run(debug=False, host="0.0.0.0", port=8000, use_reloader=False)
