
import cv2
import mediapipe as mp
import csv
import uuid
from pathlib import Path
from datetime import datetime

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

# Ask for the sign label
sign_label = input("Enter sign label (example: HELLO): ").strip().upper()

if not sign_label:
    raise ValueError("Sign label cannot be empty.")

print("\nSignCadence data collection started.")
print("Press S to save a hand capture.")
print("Press Q to quit.")

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# Open CSV file
file_exists = CSV_FILE.exists()

with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    if not file_exists:
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
            "z"
        ])

    # Start camera and detector
    with HandLandmarker.create_from_options(options) as landmarker:

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError("Could not open webcam.")

        capture_count = 0

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            # Draw detected landmarks
            for hand in detected_hands:

                for landmark in hand:

                    height, width, _ = frame.shape

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                f"Sign: {sign_label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Captures: {capture_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "S = Save | Q = Quit",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("SignCadence - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Save capture
            if key == ord("s"):

                if not detected_hands:
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
                            landmark.z
                        ])

                file.flush()

                capture_count += 1

                print(
                    f"Saved capture {capture_count} "
                    f"for sign {sign_label}"
                )

            # Exit
            elif key == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

print("\nSignCadence data collection finished.")
print(f"Saved data to: {CSV_FILE}")
