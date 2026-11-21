
import cv2
import mediapipe as mp
import csv
import uuid
from pathlib import Path
from datetime import datetime
import time

# ---------------------------------
# SIGNCADENCE - HAND DATA COLLECTION
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DATASET_FOLDER = PROJECT_ROOT / "dataset" / "landmarks"
CSV_FILE = DATASET_FOLDER / "signcadence_landmarks.csv"

DATASET_FOLDER.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

# ---------------------------------
# SIGNER INFORMATION
# ---------------------------------

signer_id = input(
    "Enter anonymous signer ID (example: TEST_SIGNER_002): "
).strip().upper()

if not signer_id:
    raise ValueError("Signer ID cannot be empty.")

sign_label = input(
    "Enter sign label (example: HELLO): "
).strip().upper()

if not sign_label:
    raise ValueError("Sign label cannot be empty.")

print("\nSignCadence data collection started.")
print("Signer:", signer_id)
print("Sign:", sign_label)
print("Click the camera window before using the keyboard.")
print("Press S to save a hand capture.")
print("Press Q to quit.")

# ---------------------------------
# MEDIAPIPE SETUP
# ---------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# ---------------------------------
# CSV SETUP
# ---------------------------------

file_exists = CSV_FILE.exists()
capture_count = 0

with open(
    CSV_FILE,
    "a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    if not file_exists or CSV_FILE.stat().st_size == 0:
        writer.writerow([
            "capture_id",
            "sign_label",
            "capture_time",
            "timestamp_ms",
            "hand_index",
            "handedness",
            "landmark_index",
            "x",
            "y",
            "z",
            "signer_id"
        ])

    # ---------------------------------
    # START HAND DETECTOR
    # ---------------------------------

    with HandLandmarker.create_from_options(options) as landmarker:

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError("Could not open webcam.")

        status_message = "Waiting for hand..."
        status_until = 0

        try:
            while True:

                success, frame = camera.read()

                if not success:
                    print("Could not read camera frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(
                    datetime.now().timestamp() * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                detected_hands = result.hand_landmarks

                height, width, _ = frame.shape

                # Draw hand landmarks
                for hand in detected_hands:
                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )

                # Display information
                display_lines = [
                    (f"Sign: {sign_label}", 35),
                    (f"Signer: {signer_id}", 70),
                    (f"Captures: {capture_count}", 105),
                    ("S = Save | Q = Quit", 140)
                ]

                for message, y_position in display_lines:
                    cv2.putText(
                        frame,
                        message,
                        (20, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                # Display save feedback
                if time.monotonic() < status_until:
                    cv2.putText(
                        frame,
                        status_message,
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                cv2.imshow(
                    "SignCadence - Data Collection",
                    frame
                )

                # Read keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key != 255:
                    print("Keyboard input received:", key)

                # Make keyboard input case-insensitive
                key_char = (
                    chr(key).lower()
                    if 0 <= key <= 127
                    else ""
                )

                # ---------------------------------
                # SAVE CAPTURE
                # ---------------------------------

                if key_char == "s":

                    if not detected_hands:
                        status_message = "No hand detected!"
                        status_until = time.monotonic() + 3
                        print("No hand detected. Capture not saved.")
                        continue

                    capture_id = str(uuid.uuid4())
                    capture_time = datetime.now().isoformat()

                    for hand_index, hand in enumerate(detected_hands):

                        handedness = "Unknown"

                        if (
                            result.handedness
                            and hand_index < len(result.handedness)
                        ):
                            handedness = (
                                result.handedness[hand_index][0].category_name
                            )

                        for landmark_index, landmark in enumerate(hand):

                            writer.writerow([
                                capture_id,
                                sign_label,
                                capture_time,
                                timestamp_ms,
                                hand_index,
                                handedness,
                                landmark_index,
                                landmark.x,
                                landmark.y,
                                landmark.z,
                                signer_id
                            ])

                    file.flush()

                    capture_count += 1

                    status_message = "CAPTURE SAVED!"
                    status_until = time.monotonic() + 3

                    print(
                        f"SUCCESS! Saved capture {capture_count} "
                        f"for sign {sign_label} "
                        f"by signer {signer_id}"
                    )

                # ---------------------------------
                # QUIT
                # ---------------------------------

                elif key_char == "q":
                    print("Stopping SignCadence...")
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()

print("\nSignCadence data collection finished.")
print(f"Saved data to: {CSV_FILE}")
