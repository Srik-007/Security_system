import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os

# BRIDGE: Ensure compatibility with Wayland/X11 on Arch Linux
# This prevents the 'plugin not found' errors common on Arch
os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- CONFIGURATION ---
MODEL_PATH = 'hand_landmarker.task'
CYAN = (255, 251, 0) # Recognition Color

def normalize_points(landmarks):
    """
    STRUCTURAL LOGIC: 
    By subtracting the Wrist (Point 0) coordinates from every other landmark,
    the 'password' stays the same regardless of where your hand is in the frame.
    This provides 'Translation Invariance'.
    """
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    return [{"x": p.x - base_x, "y": p.y - base_y} for p in landmarks]

# 1. Initialize the Tasks API Detector
if not os.path.exists(MODEL_PATH):
    print(f"[STRUCTURAL RED] ERROR: {MODEL_PATH} not found in this folder!")
    print("Please run: wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    exit()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("[LOGIC ORANGE] IDENTITY SEATER ACTIVE.")
print("[LOGIC ORANGE] PERFORM YOUR SECRET SIGN AND PRESS 'S' TO SAVE IT.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # Mirror for natural interaction
    
    # Prepare the frame for the Tasks API (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run the modern detection logic
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        # Extract the 21 landmarks of the first hand
        landmarks = detection_result.hand_landmarks[0]
        
        # Draw visual feedback: Circles on the joints
        for p in landmarks:
            x, y = int(p.x * frame.shape[1]), int(p.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, CYAN, -1)
            
        cv2.putText(frame, "READY TO SEAT (PRESS 'S')", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, CYAN, 2)

        # THE CAPTURE TRIGGER: Press 'S' to save the DNA of your hand sign
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            signature = normalize_points(landmarks)
            
            # Save the JSON identity token to the disk
            with open("master_sign.json", "w") as f:
                json.dump(signature, f)
            
            print("[DATA CYAN] IDENTITY TOKEN SEATED SUCCESSFULLY: master_sign.json")
            break
        elif key == ord('q'):
            break

    cv2.imshow("Integrated Genesis: Identity Seater", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Proper cleanup of the Logic Gate
detector.close()
cap.release()
cv2.destroyAllWindows()