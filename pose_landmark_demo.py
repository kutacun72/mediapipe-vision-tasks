import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "pose_landmarker_lite (1).task"
image_path = "images/pose-1.jpg"

mp_image = mp.Image.create_from_file(image_path)
cv_image = cv2.imread(image_path)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

# MediaPipe Pose landmark connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28),
    (28, 30), (28, 32), (30, 32)
]

with PoseLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

print("Pose landmark detection completed.")
print("Number of people detected:", len(result.pose_landmarks))

if len(result.pose_landmarks) > 0:
    h, w, _ = cv_image.shape
    landmarks = result.pose_landmarks[0]

    # Draw skeleton lines
    for start_idx, end_idx in POSE_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * w)
        y1 = int(start.y * h)
        x2 = int(end.x * w)
        y2 = int(end.y * h)

        cv2.line(cv_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

    # Draw landmark points
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(cv_image, (x, y), 5, (0, 255, 0), -1)

    cv2.imwrite("pose_landmark_skeleton_result.jpg", cv_image)
    print("Result image saved as pose_landmark_skeleton_result.jpg")