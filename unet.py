

# ########################################################
#best overlay
# import cv2 as cv
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf

# # ==========================
# # 1️⃣ SETUP & LENS LOAD
# # ==========================
# TFLITE_PATH = "iris_pure_float32.tflite"
# interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
# interpreter.allocate_tensors()
# input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()

# # Lens Load (IMREAD_UNCHANGED is a must)
# LENS_PATH = "images/lens_circular.png"
# lens_img = cv.imread(LENS_PATH, cv.IMREAD_UNCHANGED)

# if lens_img is None:
#     print(f"❌ ERROR: File nahi mili: {LENS_PATH}")
#     lens_img = np.zeros((300, 300, 4), dtype=np.uint8)
#     cv.circle(lens_img, (150, 150), 100, (255, 0, 0, 255), -1) 
# else:
#     print("✅ Lens Loaded Successfully!")

# # ==========================
# # 2️⃣ FORCE OVERLAY LOGIC
# # ==========================
# def apply_lens_force(crop, mask, lens):
#     h, w = crop.shape[:2]
#     # Resize lens to crop size
#     lens_res = cv.resize(lens, (w, h), interpolation=cv.INTER_LINEAR)
    
#     # 1. Extract Color and Alpha from Lens
#     if lens_res.shape[2] == 4:
#         lens_bgr = lens_res[:, :, :3]
#         lens_alpha = lens_res[:, :, 3].astype(float) / 255.0
#     else:
#         lens_bgr = lens_res
#         lens_alpha = np.ones((h, w), dtype=float)

#     # 2. Refine UNet Mask
#     # Thresholding ko thora loose rakha hai taaki poori iris cover ho
#     _, mask_binary = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)
#     mask_norm = mask_binary.astype(float) / 255.0
    
#     # 3. Final Alpha (In dono ka combine hona hi rang gayab karta tha)
#     # Is baar hum lens_alpha ko zyada priority dein ge
#     final_alpha = np.clip(mask_norm * 1.5, 0, 1) # Force visible
#     final_alpha = np.expand_dims(final_alpha, axis=-1)

#     # 4. Result Calculation
#     # Background ko dark karein aur lens ke asli RGB ko add karein
#     bg_part = crop.astype(float) * (1.0 - final_alpha)
#     fg_part = lens_bgr.astype(float) * final_alpha
    
#     result = cv.add(bg_part.astype(np.uint8), fg_part.astype(np.uint8))
#     return result

# # ==========================
# # 3️⃣ MEDIAPIPE & MODEL
# # ==========================
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)

# def get_unet_mask(crop):
#     img = cv.resize(crop, (384, 384))
#     img = img.astype(np.float32) / 255.0
#     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
#     interpreter.invoke()
#     pred = interpreter.get_tensor(output_details[0]['index'])[0]
#     mask = (pred > 0.5).astype(np.uint8) * 255
#     return cv.resize(mask, (crop.shape[1], crop.shape[0]))

# # ==========================
# # 4️⃣ MAIN LOOP
# # ==========================
# cap = cv.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
#     frame = cv.flip(frame, 1)
#     h, w, _ = frame.shape
#     results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    
#     if results.multi_face_landmarks:
#         landmarks = results.multi_face_landmarks[0].landmark
#         for idx in [468, 473]:
#             cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            
#             # Box size: 60-70 best hai
#             b = 65 
#             y1, y2, x1, x2 = max(0, cy-b), min(h, cy+b), max(0, cx-b), min(w, cx+b)
#             crop = frame[y1:y2, x1:x2].copy()
            
#             if crop.size > 0:
#                 mask = get_unet_mask(crop)
#                 # Force Overlay
#                 crop_final = apply_lens_force(crop, mask, lens_img)
#                 frame[y1:y2, x1:x2] = crop_final

#     cv.imshow("Fixed Lens Overlay", frame)
#     if cv.waitKey(1) & 0xFF == ord('q'): break

# cap.release()
# cv.destroyAllWindows()


###################################################

# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import time

# # ==========================
# # 1. SETUP & CONFIGURATION
# # ==========================
# TFLITE_PATH = "iris_pure_float32.tflite" 
# LENS_IMAGE_PATH = "images/lenses.png" # Apni transparent lens PNG yahan rakhein

# # Load TFLite Model
# interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# IMG_SIZE = 384 

# # Load MediaPipe
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     refine_landmarks=True, 
#     min_detection_confidence=0.6,
#     min_tracking_confidence=0.6
# )

# # Load Lens Texture (PNG with Alpha)
# lens_design = cv2.imread(LENS_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

# # ==========================
# # 2. RENDERING ENGINE (THE TRICK)
# # ==========================

# # ==========================
# # 3. PREDICTION ENGINE (YOUR STABLE LOGIC)
# # ==========================
# def predict_iris_precise(crop):
#     # CLAHE Enhancement for better distant detection
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
    
#     # High-quality Lanczos resizing for smooth mask
#     mask = (pred > 0.5).astype(np.uint8) * 255
#     mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
#     return mask
# def apply_pro_lens(crop, mask, lens_img):
#     h, w = crop.shape[:2]
#     eye_float = crop.astype(np.float32) / 255.0

#     # 1. Highlights (Chamak) preserve karna
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
#     highlights = cv2.threshold(gray, 0.6, 1.0, cv2.THRESH_BINARY)[1]
#     highlights_3ch = np.stack([highlights] * 3, axis=-1)

#     # 2. Lens Processing
#     lens_res = cv2.resize(lens_img, (w, h), interpolation=cv2.INTER_CUBIC)
    
#     # Check karein agar lens mein 4 channels hain
#     if lens_res.shape[2] == 4:
#         lens_bgr = lens_res[:, :, :3].astype(np.float32) / 255.0
#         # Lens ki apni transparency aur U-Net ka mask dono ko combine karein
#         lens_alpha = (lens_res[:, :, 3].astype(np.float32) / 255.0) * (mask / 255.0)
#     else:
#         lens_bgr = lens_res.astype(np.float32) / 255.0
#         lens_alpha = mask / 255.0

#     # Pupil Hole (Center black area)
#     cv2.circle(lens_alpha, (w//2, h//2), int(w * 0.12), 0, -1)
#     alpha_3ch = np.stack([lens_alpha] * 3, axis=-1)

#     # 3. THE VIVID PUSH (Door se dikhne ke liye)
#     # Lens ke colors ko 1.5x multiply kar rahe hain taaki texture "pop" kare
#     lens_vivid = np.clip(lens_bgr * 1.5 + highlights_3ch * 0.4, 0, 1)
    
#     # Final Composite
#     result = (lens_vivid * alpha_3ch + eye_float * (1 - alpha_3ch)) * 255
#     return result.astype(np.uint8)
# # ==========================
# # 4. MAIN EXECUTION LOOP
# # ==========================
# cap = cv2.VideoCapture(0)
# fps_start_time = time.time()
# frame_count = 0
# fps_display = "0"



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
        
#         # Dynamic box sizing based on distance
#         dist = np.linalg.norm(np.array([landmarks[468].x, landmarks[468].y]) - 
#                               np.array([landmarks[473].x, landmarks[473].y]))
#         box_size = int(dist * w * 0.7)
#         if box_size < 40: box_size = 50

#         for iris_idx in [468, 473]:
#             cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
#             x1, y1 = max(0, cx - box_size), max(0, cy - box_size)
#             x2, y2 = min(w, cx + box_size), min(h, cy + box_size)
            
#             crop = frame[y1:y2, x1:x2]
            
#             if crop.size > 0:
#                 try:
#                     # 1. Mask Prediction
#                     mask = predict_iris_precise(crop)
                    
#                     # 2. Professional Rendering
#                     crop_final = apply_pro_lens(crop, mask, lens_design)
                    
#                     # 3. Paste back to frame
#                     output_frame[y1:y2, x1:x2] = crop_final
#                 except Exception as e:
#                     pass

#     # FPS Calculation
#     frame_count += 1
#     current_time = time.time()
#     if current_time - fps_start_time > 1.0:
#         fps_display = str(int(frame_count / (current_time - fps_start_time)))
#         frame_count = 0
#         fps_start_time = current_time

#     cv2.putText(output_frame, f"FPS: {fps_display}", (30, 50), 
#                 cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

#     cv2.imshow("TTDEye Style Try-On", output_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'): break

# cap.release()
# cv2.destroyAllWindows()

import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ==========================
# 1️⃣ SETUP & LENS LOAD
# ==========================
TFLITE_PATH = "iris_pure_float32.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LENS_PATH = "images/trial.png" 
lens_img = cv.imread(LENS_PATH, cv.IMREAD_UNCHANGED)

if lens_img is None:
    print("❌ Lens file nahi mili! Dummy lens banaya ja raha hai.")
    lens_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv.circle(lens_img, (150, 150), 100, (255, 0, 0), -1)

# ==========================
# 2️⃣ FIXED TTDEYE LOGIC
# ==========================
def apply_ttdeye_effect(crop, mask, lens):
    h, w = crop.shape[:2]
    
    # A. Highlight Extraction (Original chamak save karein)
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    _, highlights = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    highlights_3ch = cv.merge([highlights]*3)
    
    # B. Lens Resizing & Channels
    lens_res = cv.resize(lens, (w, h), interpolation=cv.INTER_AREA)
    lens_bgr = lens_res[:, :, :3] if lens_res.shape[2] == 4 else lens_res

    # C. Dynamic Pupil Mask (Center 22% khali)
    pupil_mask = np.zeros((h, w), dtype=np.float32)
    cv.circle(pupil_mask, (w // 2, h // 2), int(w * 0.22), 1.0, -1)
    
    # D. Final Alpha Mixing
    unet_mask = mask.astype(float) / 255.0
    # Iris area (UNet) - Pupil Hole
    combined_alpha = unet_mask * (1.0 - pupil_mask)
    # Natural blend ke liye feathering
    combined_alpha = cv.GaussianBlur(combined_alpha, (7, 7), 0)
    alpha_3d = np.expand_dims(combined_alpha, axis=-1)

    # E. Precise Blending (Float to Uint8 conversion)
    # Background * (1-alpha) + Foreground * alpha
    bg_part = crop.astype(float) * (1.0 - alpha_3d)
    fg_part = lens_bgr.astype(float) * alpha_3d
    
    # Combine aur clip karein taaki pixels 0-255 ke bahar na jayein
    result = np.clip(bg_part + fg_part, 0, 255).astype(np.uint8)

    # F. Overlay Highlights (Realism)
    # Jahan original highlights thi, wahan brightness 30% increase karein
    result = np.where(highlights_3ch > 0, cv.addWeighted(result, 0.7, highlights_3ch, 0.3, 0), result)
    
    return result.astype(np.uint8)

# ==========================
# 3️⃣ MODEL & MEDIAPIPE
# ==========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)

def get_unet_mask(crop):
    img = cv.resize(crop, (384, 384))
    img = img.astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    # Squeeze pred agar usme extra dimensions hain
    mask = (np.squeeze(pred) > 0.5).astype(np.uint8) * 255
    return cv.resize(mask, (crop.shape[1], crop.shape[0]))

# ==========================
# 4️⃣ MAIN LOOP
# ==========================
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        for idx in [468, 473]: # Left & Right Iris centers
            cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            
            b = 60 # Box size
            y1, y2, x1, x2 = max(0, cy-b), min(h, cy+b), max(0, cx-b), min(w, cx+b)
            crop = frame[y1:y2, x1:x2].copy()
            
            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                mask = get_unet_mask(crop)
                result_eye = apply_ttdeye_effect(crop, mask, lens_img)
                # Final placement
                frame[y1:y2, x1:x2] = result_eye

    cv.imshow("TTDEye Final Fix", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()