#best code with green mask

# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import time

# # ==========================
# # 1️⃣ LOAD TFLITE MODEL
# # ==========================
# TFLITE_PATH = "iris_pure_float32.tflite" 
# interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# IMG_SIZE = 384 

# # ==========================
# # 2️⃣ MEDIAPIPE SETUP
# # ==========================
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     refine_landmarks=True, 
#     min_detection_confidence=0.6,
#     min_tracking_confidence=0.6
# )

# LEFT_IRIS_CENTER = 468
# RIGHT_IRIS_CENTER = 473

# # ==========================
# # 3️⃣ HELPERS
# # ==========================
# def get_iris_circle(mask):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None

#     largest = max(contours, key=cv2.contourArea)
#     (cx, cy), radius = cv2.minEnclosingCircle(largest)

#     return int(cx), int(cy), int(radius)

# def remove_pupil(mask, radius):
#     pupil_r = int(radius * 0.35)
#     h, w = mask.shape

#     pupil_mask = np.zeros_like(mask)
#     cv2.circle(pupil_mask, (w//2, h//2), pupil_r, 255, -1)

#     return cv2.bitwise_and(mask, cv2.bitwise_not(pupil_mask))

# def radial_alpha(size, radius):
#     h, w = size
#     y, x = np.ogrid[:h, :w]
#     cy, cx = h // 2, w // 2

#     dist = np.sqrt((x - cx)**2 + (y - cy)**2)
#     alpha = 1 - (dist / radius)
#     alpha = np.clip(alpha, 0, 1)

#     alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
#     return alpha

# def apply_contact_lens(roi, mask):
#     circle = get_iris_circle(mask)
#     if circle is None:
#         return roi

#     cx, cy, r = circle

#     mask = remove_pupil(mask, r)
#     alpha = radial_alpha(mask.shape, r)

#     # Lens color (BGR)
#     lens = np.zeros_like(roi)
#     lens[:] = (80, 120, 255)  # Blue lens

#     mask_norm = (mask / 255.0)

#     for c in range(3):
#         roi[:, :, c] = roi[:, :, c] * (1 - alpha * mask_norm) + \
#                        lens[:, :, c] * (alpha * mask_norm)

#     return roi.astype(np.uint8)

# def get_precise_crop(frame, center_landmark, w, h, box_size=70):
#     cx, cy = int(center_landmark.x * w), int(center_landmark.y * h)
#     x1, y1 = max(0, cx - box_size), max(0, cy - box_size)
#     x2, y2 = min(w, cx + box_size), min(h, cy + box_size)
#     return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# def predict_iris_precise(crop):
#     # Image Enhancement
#     lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     l = clahe.apply(l)
#     enhanced = cv2.merge((l, a, b))
#     enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
#     img = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
#     img = img.astype(np.float32) / 255.0
#     img = np.expand_dims(img, axis=0)
    
#     interpreter.set_tensor(input_details[0]['index'], img)
#     interpreter.invoke()
#     pred = interpreter.get_tensor(output_details[0]['index'])[0]
    
#     mask = (pred > 0.5).astype(np.uint8) * 255
#     mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
#     mask = cv2.GaussianBlur(mask, (7, 7), 0)
#     return mask

# # ==========================
# # 4️⃣ CAMERA & FPS SETUP (ERROR FIX HERE)
# # ==========================
# cap = cv2.VideoCapture(0) # Yeh line hona zaroori hai loop se pehle

# fps_start_time = time.time()
# frame_count = 0
# fps_display = "0"

# # ==========================
# # 5️⃣ REAL-TIME LOOP
# # ==========================
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
    
#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)
#     output_frame = frame.copy()

#     if results.multi_face_landmarks:
#         landmarks = results.multi_face_landmarks[0].landmark
#         for center_idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
#             center_lm = landmarks[center_idx]
#             crop, (x1, y1, x2, y2) = get_precise_crop(frame, center_lm, w, h, box_size=70)
            
#             if crop.size > 0:
#                 try:
#                     mask = predict_iris_precise(crop)
#                     green_overlay = np.zeros_like(crop)
#                     green_overlay[:] = [0, 200, 0]
#                     eye_with_color = cv2.addWeighted(crop, 0.7, green_overlay, 0.3, 0)
#                     crop_final = np.where(mask[:, :, None] > 100, eye_with_color, crop)
#                     output_frame[y1:y2, x1:x2] = crop_final
#                 except:
#                     pass

#     # --- STABLE FPS LOGIC ---
#     frame_count += 1
#     current_time = time.time()
#     elapsed_time = current_time - fps_start_time

#     if elapsed_time > 1.0: # Har ek second baad update
#         fps_display = str(int(frame_count / elapsed_time))
#         frame_count = 0
#         fps_start_time = current_time

#     # FPS Display
#     cv2.putText(output_frame, f"FPS: {fps_display}", (30, 50), 
#                 cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

#     cv2.imshow("Iris Segmentation", output_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque

MASK_BUFFER_SIZE = 5
left_mask_buffer = deque(maxlen=MASK_BUFFER_SIZE)
right_mask_buffer = deque(maxlen=MASK_BUFFER_SIZE)

LENS_PATH = "images/test1.png"   # change color here
lens_png = cv2.imread(LENS_PATH, cv2.IMREAD_UNCHANGED)

# ==========================
# 1️⃣ LOAD TFLITE MODEL
# ==========================
TFLITE_PATH = "iris_pure_float32.tflite" 
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = 384 

# ==========================
# 2️⃣ MEDIAPIPE SETUP
# ==========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, 
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# ==========================
# 3️⃣ HELPERS
# ==========================
def smooth_mask(mask, buffer):
    buffer.append(mask.astype(np.float32))
    avg_mask = np.mean(buffer, axis=0)
    return (avg_mask > 120).astype(np.uint8) * 255

def get_iris_circle(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)

    return int(cx), int(cy), int(radius)

def remove_pupil(mask, radius):
    pupil_r = int(radius * 0.35)
    h, w = mask.shape

    pupil_mask = np.zeros_like(mask)
    cv2.circle(pupil_mask, (w//2, h//2), pupil_r, 255, -1)

    return cv2.bitwise_and(mask, cv2.bitwise_not(pupil_mask))

def radial_alpha(size, radius):
    h, w = size
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2

    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    alpha = 1 - (dist / radius)
    alpha = np.clip(alpha, 0, 1)

    alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
    return alpha

def apply_contact_lens(roi, mask):
    circle = get_iris_circle(mask)
    if circle is None:
        return roi

    cx, cy, r = circle

    mask = remove_pupil(mask, r)
    alpha = radial_alpha(mask.shape, r)

    # Lens color (BGR)
    lens = np.zeros_like(roi)
    lens[:] = (80, 120, 255)  # Blue lens

    mask_norm = (mask / 255.0)

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - alpha * mask_norm) + \
                       lens[:, :, c] * (alpha * mask_norm)

    return roi.astype(np.uint8)


def get_precise_crop(frame, center_landmark, w, h, box_size=70):
    cx, cy = int(center_landmark.x * w), int(center_landmark.y * h)
    x1, y1 = max(0, cx - box_size), max(0, cy - box_size)
    x2, y2 = min(w, cx + box_size), min(h, cy + box_size)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def predict_iris_precise(crop):
    # Image Enhancement
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    img = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask

# ==========================
# 4️⃣ CAMERA & FPS SETUP (ERROR FIX HERE)
# ==========================
cap = cv2.VideoCapture(0) # Yeh line hona zaroori hai loop se pehle

fps_start_time = time.time()
frame_count = 0
fps_display = "0"

# ==========================
# 5️⃣ REAL-TIME LOOP
# ==========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    output_frame = frame.copy()

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        for center_idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
            center_lm = landmarks[center_idx]
            crop, (x1, y1, x2, y2) = get_precise_crop(
                frame, center_lm, w, h, box_size=70
            )

            if crop.size == 0:
                continue

            try:
                # 🔹 UNet segmentation
                 mask = predict_iris_precise(crop)

                # # 🔹 APPLY CONTACT LENS (🔥 NEW PART)
                 crop_final = apply_contact_lens(crop, mask)

                # # 🔹 Put back into frame
                 output_frame[y1:y2, x1:x2] = crop_final

            except Exception as e:
                print("Iris error:", e)

    cv2.imshow("Contact Lens Try-On", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




