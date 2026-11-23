
import csv
import uuid
import time
from pathlib import Path
from datetime import datetime

import cv2
import mediapipe as mp


# ---------------------------------
# SIGNCADENCE CONFIGURATION
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

DATASET_FOLDER = PROJECT_ROOT / "dataset" / "landmarks"
CSV_FILE = DATASET_FOLDER / "signcadence_landmarks.csv"

VOCABULARY_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "signcadence_nsl_provisional_vocabulary.csv"
)

EXPECTED_LANDMARKS = 21

CSV_COLUMNS = [
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
]

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
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        required = [
            "concept_id",
            "english_concept",
            "community_verification_status",
        ]

        if not all(
            column in (reader.fieldnames or [])
            for column in required
        ):
            raise ValueError(
                "Vocabulary CSV is missing required columns."
            )

        vocabulary = list(reader)

    if not vocabulary:
        raise ValueError(
            "Vocabulary file contains no concepts."
        )

    return vocabulary


# ---------------------------------
# VOCABULARY SELECTION
# ---------------------------------

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

choice = input(
    "\nEnter the number of the concept to select: "
).strip()

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
# SIGNER INFORMATION
# ---------------------------------

signer_id = input(
    "\nEnter anonymous signer ID: "
).strip().upper()

if not signer_id:
    raise ValueError("Signer ID cannot be empty.")


# ---------------------------------
# CHECK MEDIAPIPE MODEL
# ---------------------------------

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"MediaPipe model not found: {MODEL_PATH}"
    )


# ---------------------------------
# CSV SAFETY CHECK
# ---------------------------------

if CSV_FILE.exists() and CSV_FILE.stat().st_size > 0:

    with CSV_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as existing_file:

        reader = csv.reader(existing_file)
        existing_columns = next(reader, [])

    if existing_columns != CSV_COLUMNS:

        raise ValueError(
            "Existing landmark CSV columns do not match "
            "the expected format. No data was added."
        )


# ---------------------------------
# MEDIAPIPE SETUP
# ---------------------------------

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
)


# ---------------------------------
# CAMERA AND CSV COLLECTION
# ---------------------------------

file_exists = CSV_FILE.exists()
capture_count = 0

with CSV_FILE.open(
    "a",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    if not file_exists or CSV_FILE.stat().st_size == 0:
        writer.writerow(CSV_COLUMNS)
        file.flush()

    print("\nSignCadence collection started.")
    print("Sign:", sign_label)
    print("Signer:", signer_id)
    print("One hand required.")
    print("Exactly 21 landmarks required.")
    print("Press S to save. Press Q to quit.")

    with HandLandmarker.create_from_options(options) as landmarker:

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            camera.release()
            raise RuntimeError("Could not open webcam.")

        status_message = "Waiting for hand..."
        status_until = 0

        last_timestamp_ms = 0

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

                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    last_timestamp_ms + 1,
                )

                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                detected_hands = result.hand_landmarks

                height, width, _ = frame.shape

                # Draw detected landmarks.

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

                # Display information.

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

                # ---------------------------------
                # SAVE CAPTURE
                # ---------------------------------

                if key_char == "s":

                    if len(detected_hands) != 1:

                        status_message = (
                            "Exactly one hand required!"
                        )

                        status_until = time.monotonic() + 3

                        print(
                            "Capture rejected. "
                            "Exactly one hand is required."
                        )

                        continue

                    hand = detected_hands[0]

                    if len(hand) != EXPECTED_LANDMARKS:

                        status_message = (
                            "Invalid landmark count!"
                        )

                        status_until = time.monotonic() + 3

                        print(
                            "Capture rejected. "
                            "Expected exactly 21 landmarks."
                        )

                        continue

                    capture_id = str(uuid.uuid4())

                    capture_time = datetime.now().isoformat()

                    handedness = "Unknown"

                    if (
                        result.handedness
                        and len(result.handedness) > 0
                    ):

                        handedness = (
                            result.handedness[0][0].category_name
                        )

                    # Prepare all rows before writing.

                    capture_rows = []

                    for landmark_index, landmark in enumerate(hand):

                        capture_rows.append([
                            capture_id,
                            sign_label,
                            capture_time,
                            timestamp_ms,
                            0,
                            handedness,
                            landmark_index,
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            signer_id,
                        ])

                    # Final safety check.

                    if len(capture_rows) != EXPECTED_LANDMARKS:

                        print(
                            "Capture rejected. "
                            "Incomplete landmark data."
                        )

                        continue

                    # Save the complete capture together.

                    writer.writerows(capture_rows)

                    file.flush()

                    capture_count += 1

                    status_message = "CAPTURE SAVED!"

                    status_until = time.monotonic() + 3

                    print(
                        f"Saved capture {capture_count}: "
                        f"{sign_label} ({signer_id})"
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


# ---------------------------------
# COLLECTION COMPLETE
# ---------------------------------

print("\nSignCadence collection finished.")

print("New captures saved:", capture_count)

print("Saved data to:", CSV_FILE)
