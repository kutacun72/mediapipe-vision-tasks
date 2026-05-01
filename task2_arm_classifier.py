import sys
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker_lite (1).task"

# MediaPipe Pose landmark indexes
LEFT_SHOULDER = 11
LEFT_WRIST = 15
RIGHT_SHOULDER = 12
RIGHT_WRIST = 16


def classify_arm_up(image_path):
    mp_image = mp.Image.create_from_file(image_path)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if len(result.pose_landmarks) == 0:
        return "None"

    landmarks = result.pose_landmarks[0]

    left_shoulder = landmarks[LEFT_SHOULDER]
    left_wrist = landmarks[LEFT_WRIST]

    right_shoulder = landmarks[RIGHT_SHOULDER]
    right_wrist = landmarks[RIGHT_WRIST]

    left_arm_up = left_wrist.y < left_shoulder.y
    right_arm_up = right_wrist.y < right_shoulder.y

    if left_arm_up and right_arm_up:
        return "both"
    elif left_arm_up:
        return "left"
    elif right_arm_up:
        return "right"
    else:
        return "None"


if len(sys.argv) < 2:
    print("Usage: python task2_arm_classifier.py path_to_image")
else:
    image_path = sys.argv[1]
    result = classify_arm_up(image_path)
    print(result)