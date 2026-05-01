import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"
image_path = "images/hand-1.jpg"

mp_image = mp.Image.create_from_file(image_path)
cv_image = cv2.imread(image_path)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

with HandLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

print("Hand landmark detection completed.")
print("Number of hands detected:", len(result.hand_landmarks))

if len(result.hand_landmarks) > 0:
    h, w, _ = cv_image.shape

    for hand_landmarks in result.hand_landmarks:
        # Draw hand skeleton lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]

            x1 = int(start.x * w)
            y1 = int(start.y * h)
            x2 = int(end.x * w)
            y2 = int(end.y * h)

            cv2.line(cv_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # Draw hand landmark points
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(cv_image, (x, y), 7, (0, 255, 0), -1)

    cv2.imwrite("hand_landmark_result.jpg", cv_image)
    print("Result image saved as hand_landmark_result.jpg")
else:
    print("No hand landmarks detected.")