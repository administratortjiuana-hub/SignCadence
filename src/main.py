
import cv2
import mediapipe as mp
import csv
from pathlib import Path

# Where our landmark data will be saved
OUTPUT_FILE = Path("dataset/landmarks/test_landmarks.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Prepare MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Prepare CSV file
file_exists = OUTPUT_FILE.exists()

with open(OUTPUT_FILE, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)

    if not file_exists:
        writer.writerow([
            "capture_id",
            "timestamp_ms",
            "hand_index",
            "handedness",
            "landmark_index",
            "x",
            "y",
            "z"
        ])

    with HandLandmarker.create_from_options(options) as detector:

        camera = cv2.VideoCapture(0)
        timestamp = 0
        capture_id = 0

        if not camera.isOpened():
            print("Could not access camera.")
            exit()

        print("SignCadence landmark collection is running!")
        print("Press S to save hand coordinates.")
        print("Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp += 33

            result = detector.detect_for_video(
                mp_image, timestamp
            )

            height, width, _ = frame.shape

            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            cv2.imshow(
                "SignCadence - Landmark Collection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Press S to save the current hand coordinates
            if key == ord("s"):

                for hand_index, hand_landmarks in enumerate(
                    result.hand_landmarks
                ):

                    handedness = "Unknown"

                    if (
                        hand_index < len(result.handedness)
                        and result.handedness[hand_index]
                    ):
                        handedness = (
                            result.handedness[hand_index][0].category_name
                        )

                    for landmark_index, landmark in enumerate(
                        hand_landmarks
                    ):

                        writer.writerow([
                            capture_id,
                            timestamp,
                            hand_index,
                            handedness,
                            landmark_index,
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                csvfile.flush()
                capture_id += 1

                print(
                    f"Saved capture {capture_id} "
                    f"to {OUTPUT_FILE}"
                )

            if key == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

print("Landmark collection completed!")
