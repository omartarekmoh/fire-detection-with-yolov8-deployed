from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from process import _model

app = Flask(__name__)

camera_frames = {}  # Dictionary to store camera frames
model = YOLO("yolo/best.pt")
model.to("cuda")


def generate_frames():
    while True:
        for device_id, frame in camera_frames.items():
            # Resize frame to a smaller resolution (e.g., 640x480)
            # frame, fire, x, y = model2(model, frame)

            # Encode frame as JPEG with compression quality (0-100)
            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                50,
            ]  # Adjust quality as needed
            _, buffer = cv2.imencode(".jpg", frame, encode_param)
            frame_bytes = buffer.tobytes()

            # Yield the frame in byte format
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"


@app.route("/")
def index():
    return render_template("new.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/receive_frames/<device_id>", methods=["POST"])
def receive_frames(device_id):
    frame_data = request.data

    # Convert received data to numpy array
    nparr = np.frombuffer(frame_data, np.uint8)

    # Decode numpy array as image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 480))
    result = model(img)
    frame, fire, x, y = _model(result, img)

    # Store the received frame for the corresponding device
    camera_frames[device_id] = frame

    return jsonify({"fire": fire, "x": x, "y": y})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000, use_reloader=False)
