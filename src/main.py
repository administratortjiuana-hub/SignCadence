
import cv2
import mediapipe as mp
import csv
import uuid
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------
# SIGNCADENCE CONFIGURATION
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

DATASET_FOLDER = PROJECT_ROOT / "dataset" / "landmarks"
CSV_FILE = DATASET_FOLDER / "signcadence_landmarks.csv"

VOCABULARY_FILE = (
    PROJECT_ROOT / "dataset" / "signcadence_nsl_provisional_vocabulary.csv"
)

DATASET_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------------------------
# LOAD VOCABULARY
# ---------------------------------

def load_vocabulary():
    if not VOCABULARY_FILE.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {VOCABULARY_FILE}"
        )

    with VOCABULARY_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        required = [
            "concept_id",
            "english_concept",
            "community_verification_status",
        ]

        if not all(column in (reader.fieldnames or []) for column in required):
            raise ValueError("Vocabulary CSV is missing required columns.")

        vocabulary = list(reader)

    if not vocabulary:
        raise ValueError("Vocabulary file contains no concepts.")

    return vocabulary


vocabulary = load_vocabulary()

print("\nSIGNCADENCE - VOCABULARY SELECTION")
print("=" * 45)

for number, item in enumerate(vocabulary, start=1):
    print(
        f"{number}. {item['english_concept']} "
        f"({item['concept_id']})"
    )
    print(
        f"   Status: {item['community_verification_status']}"
    )

print("=" * 45)

choice = input("\nEnter the number of the concept to select: ").strip()

if not choice.isdigit() or not 1 <= int(choice) <= len(vocabulary):
    raise ValueError("Invalid vocabulary selection.")

selected = vocabulary[int(choice) - 1]

concept_id = selected["concept_id"].strip()
concept_name = selected["english_concept"].strip()

sign_label = "_".join(concept_name.upper().split())

verification_status = (
    selected["community_verification_status"].strip().upper()
)

print("\nSelected concept:", concept_name)
print("Concept ID:", concept_id)
print("Verification:", verification_status)


# ---------------------------------
# SIGNER INFORMATION
# ---------------------------------

signer_id = input(
    "\nEnter anonymous signer ID: "
).strip().upper()

if not signer_id:
    raise ValueError("Signer ID cannot be empty.")


# ---------------------------------
# VERIFICATION SAFEGUARD
# ---------------------------------

if not verification_status.startswith("VERIFIED"):
    print("\nCOLLECTION NOT AUTHORISED.")
    print("This concept has not been verified by a fluent NSL signer.")
    print("No training data has been saved.")
    print("\nPlease obtain community verification before collecting")
    print("official NSL training examples.")
    raise SystemExit(0)


# ---------------------------------
# CHECK MEDIAPIPE MODEL
# ---------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: {MODEL_PATH}"
    )


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
    num_hands=1,
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
    encoding="utf-8",
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
            "signer_id",
        ])

    print("\nSignCadence collection started.")
    print("Sign:", sign_label)
    print("Signer:", signer_id)
    print("Press S to save. Press Q to quit.")

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
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    datetime.now().timestamp() * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                detected_hands = result.hand_landmarks

                height, width, _ = frame.shape

                for hand in detected_hands:
                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1,
                        )

                display_lines = [
                    (f"Sign: {sign_label}", 35),
                    (f"Signer: {signer_id}", 70),
                    (f"Captures: {capture_count}", 105),
                    ("S = Save | Q = Quit", 140),
                ]

                for message, y_position in display_lines:
                    cv2.putText(
                        frame,
                        message,
                        (20, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                if time.monotonic() < status_until:
                    cv2.putText(
                        frame,
                        status_message,
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow(
                    "SignCadence - Data Collection",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                key_char = (
                    chr(key).lower()
                    if 0 <= key <= 127
                    else ""
                )

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
                                signer_id,
                            ])

                    file.flush()
                    capture_count += 1

                    status_message = "CAPTURE SAVED!"
                    status_until = time.monotonic() + 3

                    print(
                        f"Saved capture {capture_count}: "
                        f"{sign_label} ({signer_id})"
                    )

                elif key_char == "q":
                    print("Stopping SignCadence...")
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()


print("\nSignCadence collection finished.")
print("Saved data to:", CSV_FILE)
