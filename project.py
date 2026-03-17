import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os
import math
import time
import subprocess
import sys

# --- SYSTEM PATHS ---
BASE_DIR = '/home/srik/coding/vision_gate/venv'
HAND_MODEL = os.path.join(BASE_DIR, 'hand_landmarker.task')
FACE_MODEL = os.path.join(BASE_DIR, 'face_landmarker.task')
HAND_KEY = os.path.join(BASE_DIR, 'master_sign.json')
FACE_KEYS = os.path.join(BASE_DIR, 'face_keys.json')

# --- RECOGNITION THRESHOLDS ---
HAND_THRESH = 0.06
FACE_THRESH = 0.08 
CONFIRMATION_FRAMES = 5 

def get_error(live, master):
    error = 0
    for i in range(len(live)):
        dx = live[i]['x'] - master[i]['x']
        dy = live[i]['y'] - master[i]['y']
        error += math.sqrt(dx**2 + dy**2)
    return error / len(live)

def is_locked():
    """Returns True only if hyprlock is actually running."""
    try:
        subprocess.check_output(["pgrep", "hyprlock"])
        return True
    except subprocess.CalledProcessError:
        return False

# 1. Load Biometric Databases
try:
    with open(HAND_KEY, "r") as f: hand_master = json.load(f)
    with open(FACE_KEYS, "r") as f: face_masters = json.load(f)
except Exception as e:
    print(f"[ERROR] Missing Biometric Files: {e}")
    sys.exit(1)

# 2. Pre-Initialize AI Models
base_hand = python.BaseOptions(model_asset_path=HAND_MODEL)
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=base_hand, num_hands=1)
)

base_face = python.BaseOptions(model_asset_path=FACE_MODEL)
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base_face, num_faces=1)
)

print("[SENTINEL] STANDING BY. Camera is currently OFF.")

while True:
    if is_locked():
        print("[SENTINEL] LOCK DETECTED. SENSORS ONLINE.")
        cap = cv2.VideoCapture(0)
        
        # Recognition Loop
        while is_locked() and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            hand_res = hand_detector.detect(mp_image)
            face_res = face_detector.detect(mp_image)
            
            access_granted = False
            
            if hand_res.hand_landmarks:
                h_live = [{"x": p.x - hand_res.hand_landmarks[0][0].x, "y": p.y - hand_res.hand_landmarks[0][0].y} for p in hand_res.hand_landmarks[0]]
                if get_error(h_live, hand_master) < HAND_THRESH:
                    access_granted = True

            if not access_granted and face_res.face_landmarks:
                f_live = [{"x": p.x - face_res.face_landmarks[0][1].x, "y": p.y - face_res.face_landmarks[0][1].y} for p in face_res.face_landmarks[0]]
                for master in face_masters:
                    if get_error(f_live, master) < FACE_THRESH:
                        access_granted = True
                        break

            if access_granted:
                print("[SENTINEL] MATCH FOUND. EXECUTING UNLOCK.")
                # Force unlock and kill hyprlock
                os.system("loginctl unlock-session")
                os.system("killall -9 hyprlock")
                
                # IMPORTANT: Break the inner loop immediately
                break 
            
            time.sleep(0.05)
        
        # CLEANUP PHASE
        cap.release()
        cv2.destroyAllWindows()
        print("[SENTINEL] ACCESS GRANTED. SENSORS OFFLINE.")
        
        # Grace period: Wait 5 seconds before checking for a lock again.
        # This prevents the script from instantly re-opening the camera.
        time.sleep(5) 
    
    # Low power polling while unlocked
    time.sleep(2)