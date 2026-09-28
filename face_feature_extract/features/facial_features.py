# pip install face_recognition pillow numpy

import face_recognition
from PIL import Image, ImageDraw
import numpy as np

def extract_facial_features(image_path):
    # Load image
    image = face_recognition.load_image_file(image_path)
    
    # Get facial landmarks for all faces
    landmarks_list = face_recognition.face_landmarks(image)
    
    if not landmarks_list:
        print("No faces found in the image.")
        return
    
    print(f"Found {len(landmarks_list)} face(s)\n")
    
    for i, landmarks in enumerate(landmarks_list):
        print(f"── Face {i + 1} ──────────────────────")
        for feature, points in landmarks.items():
            print(f"  {feature:<20} → {len(points)} points: {points[:2]}...")
    
    return landmarks_list


def draw_features(image_path, output_path="output.png"):
    image = face_recognition.load_image_file(image_path)
    landmarks_list = face_recognition.face_landmarks(image)
    
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Color for each facial feature
    colors = {
        "chin":              "white",
        "left_eyebrow":      "yellow",
        "right_eyebrow":     "yellow",
        "nose_bridge":       "cyan",
        "nose_tip":          "cyan",
        "left_eye":          "violet",
        "right_eye":         "violet",
        "top_lip":           "lime",
        "bottom_lip":        "lime",
    }
    
    for landmarks in landmarks_list:
        for feature, points in landmarks.items():
            color = colors.get(feature, "white")
            # Draw lines connecting the feature points
            draw.line(points, fill=color, width=2)
            # Draw dots at each point
            for (x, y) in points:
                draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
    
    pil_image.save(output_path)
    print(f"\nSaved annotated image → {output_path}")
    pil_image.show()


# ── Run ──────────────────────────────────────────────────────────────
image_path = "image.jpg"   # ← change this to your image path

features = extract_facial_features(image_path)
draw_features(image_path)