import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "face_landmarker.task"
image_path = "images/face-1.png"

mp_image = mp.Image.create_from_file(image_path)
cv_image = cv2.imread(image_path)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# Basic face mesh connection groups
FACE_CONNECTIONS = [
    # Face oval / jaw outline
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
    (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
    (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
    (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
    (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
    (109, 10),

    # Left eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (153, 154), (154, 155), (155, 133), (133, 173), (173, 157),
    (157, 158), (158, 159), (159, 160), (160, 161), (161, 246),
    (246, 33),

    # Right eye
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
    (380, 381), (381, 382), (382, 362), (362, 398), (398, 384),
    (384, 385), (385, 386), (386, 387), (387, 388), (388, 466),
    (466, 263),

    # Lips outer
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (291, 308), (308, 324), (324, 318), (318, 402), (402, 317),
    (317, 14), (14, 87), (87, 178), (178, 88), (88, 95),
    (95, 61),

    # Nose bridge and nose bottom
    (168, 6), (6, 197), (197, 195), (195, 5),
    (5, 4), (4, 45), (4, 275), (45, 220), (275, 440)
]

with FaceLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

print("Face landmark detection completed.")
print("Number of faces detected:", len(result.face_landmarks))

if len(result.face_landmarks) > 0:
    h, w, _ = cv_image.shape
    landmarks = result.face_landmarks[0]

    # Draw face mesh lines
    for start_idx, end_idx in FACE_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * w)
        y1 = int(start.y * h)
        x2 = int(end.x * w)
        y2 = int(end.y * h)

        cv2.line(cv_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Draw landmark points
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(cv_image, (x, y), 1, (0, 255, 0), -1)

    cv2.imwrite("face_landmark_mesh_result.jpg", cv_image)
    print("Result image saved as face_landmark_mesh_result.jpg")