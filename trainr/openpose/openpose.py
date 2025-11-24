import os
import cv2 as cv
import numpy as np
from flask import Flask, Response
import time

app = Flask(__name__)

# Define body parts and pose pairs
BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
    "Background": 18,
    "UpperBack": 19,
    "MiddleBack": 20,
    "LowerBack": 21,
}

POSE_PAIRS = [
    ["Neck", "RShoulder"],
    ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["Neck", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["Neck", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
    ["Nose", "REye"],
    ["REye", "REar"],
    ["Nose", "LEye"],
    ["LEye", "LEar"],
    ["Neck", "UpperBack"],
    ["UpperBack", "MiddleBack"],
    ["MiddleBack", "LowerBack"],
]

# Load the neural network (use absolute path relative to this file)
model_path = os.path.join(os.path.dirname(__file__), "graph_opt.pb")
net = cv.dnn.readNetFromTensorflow(model_path)

# Variables for scoring
green_border_count = 0
red_border_count = 0
start_time = time.time()
last_score_time = start_time  # Tracks time of last score update
current_score = 0  # Initialize current score

# Function to generate video frames
def generate_frames():
    global green_border_count, red_border_count, last_score_time, current_score
    cap = cv.VideoCapture(0)  # 0 for the default camera

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        # Resize frame as per requirements
        inWidth, inHeight = 368, 368
        net.setInput(
            cv.dnn.blobFromImage(
                frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False
            )
        )
        out = net.forward()
        out = out[:, :22, :, :]

        frameWidth, frameHeight = frame.shape[1], frame.shape[0]
        points = []

        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > 0.2 else None)

        # Calculate the elbow and shoulder angles
        elbow_angle_ok = shoulder_angle_ok = False
        if (
            points[BODY_PARTS["RShoulder"]] and points[BODY_PARTS["RElbow"]] and points[BODY_PARTS["RWrist"]]
        ):
            shoulder = np.array(points[BODY_PARTS["RShoulder"]])
            elbow = np.array(points[BODY_PARTS["RElbow"]])
            wrist = np.array(points[BODY_PARTS["RWrist"]])

            upper_arm = elbow - shoulder
            forearm = wrist - elbow
            angle_rad = np.arctan2(forearm[1], forearm[0]) - np.arctan2(upper_arm[1], upper_arm[0])
            angle_deg = np.degrees(angle_rad)
            angle_deg = angle_deg + 360 if angle_deg < 0 else angle_deg
            angle_deg = 360 - angle_deg if angle_deg > 180 else angle_deg

            elbow_angle_ok = 0 <= angle_deg <= 120
            cv.putText(
                frame, f"Elbow Angle: {int(angle_deg)}", (points[BODY_PARTS["RElbow"]][0] + 10, points[BODY_PARTS["RElbow"]][1]),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
            )

        if (
            points[BODY_PARTS["Neck"]] and points[BODY_PARTS["RShoulder"]] and points[BODY_PARTS["RElbow"]]
        ):
            neck = np.array(points[BODY_PARTS["Neck"]])
            shoulder = np.array(points[BODY_PARTS["RShoulder"]])
            elbow = np.array(points[BODY_PARTS["RElbow"]])

            neck_to_shoulder = neck - shoulder
            shoulder_to_elbow = elbow - shoulder
            angle_rad = np.arctan2(shoulder_to_elbow[1], shoulder_to_elbow[0]) - np.arctan2(neck_to_shoulder[1], neck_to_shoulder[0])
            shoulder_angle_deg = np.degrees(angle_rad)
            shoulder_angle_deg = shoulder_angle_deg + 360 if shoulder_angle_deg < 0 else shoulder_angle_deg
            shoulder_angle_deg = 360 - shoulder_angle_deg if shoulder_angle_deg > 180 else shoulder_angle_deg

            shoulder_angle_ok = 70 <= shoulder_angle_deg <= 120
            cv.putText(
                frame, f"Shoulder Angle: {int(shoulder_angle_deg)}",
                (points[BODY_PARTS["RShoulder"]][0] + 10, points[BODY_PARTS["RShoulder"]][1] + 20),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
            )

        # Determine border color
        if elbow_angle_ok and shoulder_angle_ok:
            border_color = (0, 255, 0)
            green_border_count += 1
        else:
            border_color = (0, 0, 255)
            red_border_count += 1

        cv.rectangle(frame, (0, 0), (frameWidth - 1, frameHeight - 1), border_color, 10)

        # Update score every 5 seconds
        current_score = 0
        current_time = time.time()
        if current_time - last_score_time >= 5:
            
            current_score = current_score+ 1 if green_border_count > red_border_count else current_score
            print(f"Score Update: {current_score} (Green borders: {green_border_count}, Red borders: {red_border_count})")
            green_border_count = 0
            red_border_count = 0
            last_score_time = current_time  # Update last score time

        # Display score on the top-left corner of the frame
        cv.putText(
            frame,
            f"Score: {current_score}",
            (10, 30),  # Top-left position
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),  # Black color
            2,
        )

        # Draw the body parts and connections
        for pair in POSE_PAIRS:
            partFrom, partTo = pair[0], pair[1]
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        # Encode the frame as JPEG
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the format required for video streaming
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video-feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(port=5001)  # Adjust port as needed
