import cvzone
import cv2
import math


def _model(result, frame):
    classnames = ["Fire"]
    x1, y1, x2, y2 = 0, 0, 0, 0
    fire = False
    for info in result:

        boxes = info.boxes
        for box in boxes:
            fire = True
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 30:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(
                    frame,
                    f"{classnames[Class]} {confidence}%",
                    [x1 + 8, y1 + 100],
                    scale=1.5,
                    thickness=2,
                )

    return frame, fire, (x1 + x2) / 2, (y1 + y2) / 2
