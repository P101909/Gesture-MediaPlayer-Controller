"""
Gesture-Based Media Controller

This script lets me control a media player using hand gestures through my webcam.
I use MediaPipe's hand landmarker and my own gesture-recognition .task model.
Each gesture triggers a keyboard shortcut, like play/pause or volume control.
I also added gesture smoothing and on-screen labels to make the system usable.
"""

import cv2
import time
from pynput.keyboard import Controller, Key
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


# Path to my custom gesture-recognizer model
GESTURE_MODEL = "/Users/kush/Documents/SFU_CMPT/FALL2025/Gesture_Project/cmpt310/cmpt310.git/cmpt310/models/gesture_recognizer.task"

# Path to MediaPipe’s hand landmark model
"""
The official MediaPipe hand_landmarker.task file can be downloaded from the MediaPipe models repository.
The specific path to the model is:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

"""
HAND_MODEL = "/Users/kush/Documents/SFU_CMPT/FALL2025/Gesture_Project/cmpt310/cmpt310.git/cmpt310/models/hand_landmarker.task"

# Used to send keyboard shortcuts to the computer
keyboard = Controller()

# Mapping from detected gesture → what action I want to perform
MEDIA_CONTROLS = {
    "thumbs_up": "play_pause",
    "fist": "volume_down",
    "victory": "volume_up",
    "swipe_left": "rewind",
    "swipe_right": "forward",
    "open_palm": "exit_app"
}

# Colors used to draw gesture labels on the screen
GESTURE_COLORS = {
    "thumbs_up":  (0, 255, 0),
    "fist":       (0, 0, 255),
    "victory":    (255, 0, 0),
    "swipe_left": (0, 128, 255),
    "swipe_right":(0, 128, 255),
    "open_palm":  (200, 200, 200),
    "none":       (100, 100, 100)
}

# These variables help me stabilize the gesture prediction
LAST_GESTURE = "none"
STABLE_GESTURE = "none"
GESTURE_FRAMES = 0
CONFIRMATION_FRAMES = 6   # the gesture must stay the same for 6 frames


# Loading the gesture recognizer model
gesture_options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path=GESTURE_MODEL),
    running_mode=vision.RunningMode.IMAGE
)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

# Loading the hand landmark model
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)


# This function performs the keyboard actions for each gesture
def trigger_media_action(action):
    if action == "play_pause":
        keyboard.press(" "); keyboard.release(" ")
    elif action == "volume_up":
        keyboard.press(Key.media_volume_up); keyboard.release(Key.media_volume_up)
    elif action == "volume_down":
        keyboard.press(Key.media_volume_down); keyboard.release(Key.media_volume_down)
    elif action == "rewind":
        keyboard.press(Key.left); keyboard.release(Key.left)
    elif action == "forward":
        keyboard.press(Key.right); keyboard.release(Key.right)
    elif action == "exit_app":
        print("Exit gesture detected. Closing...")
        return "EXIT"
    return None


# Start reading the webcam feed
cap = cv2.VideoCapture(0)
prev_time = time.time()

print("Running… press 'q' to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera error.")
        break

    # MediaPipe needs RGB, so I convert from OpenCV’s BGR format
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    # Detect the hand and landmarks
    hand_result = hand_detector.detect(mp_image)
    x1 = y1 = x2 = y2 = None

    if hand_result.hand_landmarks:
        h, w, _ = frame.shape
        lm = hand_result.hand_landmarks[0]

        # Convert normalized landmark positions into pixel coordinates
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]

        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Draw the bounding box around the hand
        cv2.rectangle(frame, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 2)

        # Draw each landmark point
        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 255), -1)

    # Detect the gesture for the current frame
    gesture_result = gesture_recognizer.recognize(mp_image)

    raw_gesture = "none"
    confidence = 0

    # Keep only the top predicted gesture
    if gesture_result.gestures:
        top = gesture_result.gestures[0][0]
        raw_gesture = top.category_name
        confidence = top.score

    # Smoothing logic to avoid predictions flickering too fast
    if raw_gesture == LAST_GESTURE:
        GESTURE_FRAMES += 1
    else:
        GESTURE_FRAMES = 0

    if GESTURE_FRAMES >= CONFIRMATION_FRAMES:
        STABLE_GESTURE = raw_gesture

    LAST_GESTURE = raw_gesture

    # Trigger a media action when a gesture becomes stable
    if STABLE_GESTURE in MEDIA_CONTROLS:
        action = MEDIA_CONTROLS[STABLE_GESTURE]
        result = trigger_media_action(action)
        if result == "EXIT":
            break

    # Draw the recognized gesture on the screen
    color = GESTURE_COLORS.get(STABLE_GESTURE, (255, 255, 255))

    if x1 is not None:
        label_y = y1 - 30 if y1 - 30 > 0 else y1 + 30

        cv2.rectangle(frame,
                      (x1, label_y - 25),
                      (x1 + 250, label_y + 5),
                      (0, 0, 0), -1)

        cv2.putText(frame, f"{STABLE_GESTURE} ({confidence:.2f})",
                    (x1 + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Calculate and draw FPS
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    # Show everything in a window
    cv2.imshow("Gesture Control – Enhanced", frame)

    # Press 'q' to quit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()