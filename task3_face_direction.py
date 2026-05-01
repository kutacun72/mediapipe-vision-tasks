import sys
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"

# Face landmark indexes
NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454


def classify_face_direction(image_path):
    mp_image = mp.Image.create_from_file(image_path)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if len(result.face_landmarks) == 0:
        return "straight"

    landmarks = result.face_landmarks[0]

    nose_x = landmarks[NOSE_TIP].x
    left_x = landmarks[LEFT_FACE].x
    right_x = landmarks[RIGHT_FACE].x

    face_center_x = (left_x + right_x) / 2

    difference = nose_x - face_center_x

    threshold = 0.03

    if difference < -threshold:
        return "left"
    elif difference > threshold:
        return "right"
    else:
        return "straight"


if len(sys.argv) < 2:
    print("Usage: python task3_face_direction.py path_to_image")
else:
    image_path = sys.argv[1]
    result = classify_face_direction(image_path)
    print(result)