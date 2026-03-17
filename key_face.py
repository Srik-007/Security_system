import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os

# --- CONFIGURATION ---
BASE_DIR = '/home/srik/coding/vision_gate/venv'
MODEL_PATH = os.path.join(BASE_DIR, 'face_landmarker.task')
SAVE_PATH = os.path.join(BASE_DIR, 'face_keys.json')

def normalize_face(landmarks):
    # Use the nose tip (index 1) as the anchor point
    anchor = landmarks[1]
    return [{"x": p.x - anchor.x, "y": p.y - anchor.y, "z": p.z - anchor.z} for p in landmarks]

# Initialize Face Landmarker
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found. Download it first!")
    exit()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=False, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
face_database = []

print("--- FACE IDENTITY SEATER ---")
print("Press 'S' to save an angle. Press 'Q' to finish and save database.")
print("Angles to capture: Front, Left, Right, Up, Down.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    res = detector.detect(mp_image)

    if res.face_landmarks:
        landmarks = res.face_landmarks[0]
        # Draw some landmarks for feedback
        for i in [1, 33, 263, 61, 291]: # Nose, Eyes, Mouth corners
            p = landmarks[i]
            x, y = int(p.x * frame.shape[1]), int(p.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            face_database.append(normalize_face(landmarks))
            print(f"Angle {len(face_database)} seated.")
        elif key == ord('q'):
            break

    cv2.imshow("Face Seater", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Save the multi-angle database
with open(SAVE_PATH, "w") as f:
    json.dump(face_database, f)

print(f"Database saved to {SAVE_PATH}. Seating complete.")
detector.close()
cap.release()
cv2.destroyAllWindows()