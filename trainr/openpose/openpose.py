import cv2 as cv
import numpy as np
from flask import Flask, Response
import time

app = Flask(__name__)

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

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

# Variables for scoring
green_border_count = 0
red_border_count = 0
start_time = time.time()
last_score_time = start_time  # Tracks time of last score update
current_score = 0  # Initialize current score
net = None  # Will be loaded lazily

def create_error_frame(message):
    """Create an error frame image"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv.putText(frame, message, (50, 240), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(frame, "Check console for details", (50, 280), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ret, buffer = cv.imencode('.jpg', frame)
    return buffer.tobytes()

# Load the neural network (lazy loading)
def load_network():
    global net
    if net is None:
        try:
            import os
            model_path = "graph_opt.pb"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"Loading model from {model_path}...")
            net = cv.dnn.readNetFromTensorflow(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return net

# Function to generate video frames
def generate_frames():
    global green_border_count, red_border_count, last_score_time, current_score
    
    # Load network on first use
    try:
        net = load_network()
    except Exception as e:
        error_frame = create_error_frame(f"Model Error: {str(e)}")
        while True:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)
    
    # Try to open camera
    cap = None
    try:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")
        # Set camera properties for better performance
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera opened successfully!")
    except Exception as e:
        print(f"Error opening camera: {e}")
        error_frame = create_error_frame(f"Camera Error: {str(e)}")
        while True:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)
    
    try:
        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print("Failed to read frame from camera")
                break

            # Resize frame as per requirements
            inWidth, inHeight = 368, 368
            try:
                net.setInput(
                    cv.dnn.blobFromImage(
                        frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False
                    )
                )
                out = net.forward()
                out = out[:, :22, :, :]
            except Exception as e:
                print(f"Error in neural network processing: {e}")
                # Continue with next frame
                continue

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
            current_time = time.time()
            if current_time - last_score_time >= 5:
                current_score = current_score + 1 if green_border_count > red_border_count else current_score
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
    except Exception as e:
        print(f"Error in frame generation: {e}")
        error_frame = create_error_frame(f"Processing Error: {str(e)}")
        while True:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)
    finally:
        if cap is not None:
            cap.release()
            print("Camera released")

@app.route('/video-feed')
def video_feed():
    return Response(
        generate_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

if __name__ == "__main__":
    app.run(port=5001)  # Adjust port as needed
