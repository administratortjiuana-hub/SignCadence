
import cv2
import mediapipe as mp
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime


# ---------------------------------
# SIGNCADENCE LIVE RECOGNITION
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "signcadence_model.joblib"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained AI model not found. Train a model before recognition."
    )

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe hand model not found: {HAND_MODEL_PATH}"
    )


# ---------------------------------
# LOAD TRAINED CLASSIFIER
# ---------------------------------

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
feature_columns = saved_data["feature_columns"]
labels = saved_data["labels"]

print("\nSIGNCADENCE LIVE RECOGNITION")
print("Available labels:", labels)


# ---------------------------------
# MEDIAPIPE SETUP
# ---------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
)


# ---------------------------------
# CAMERA
# ---------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("\nCamera started.")
    print("Show one hand to the camera.")
    print("Press Q to quit.")

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print("Camera frame could not be read.")
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

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                features = []

                for landmark in hand:
                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                prediction = model.predict([features])[0]

                probabilities = model.predict_proba([features])[0]
                confidence = float(np.max(probabilities))

                display_text = (
                    f"{prediction} - {confidence:.1%}"
                )

                cv2.putText(
                    frame,
                    display_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                height, width, _ = frame.shape

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

            else:

                cv2.putText(
                    frame,
                    "Show your hand",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(
                "SignCadence - Live Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

print("\nSignCadence recognition stopped.")
