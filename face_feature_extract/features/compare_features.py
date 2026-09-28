import face_recognition
import numpy as np
import json
import os
from PIL import Image, ImageDraw

# ─── Database file to store encodings ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE  = os.path.join(BASE_DIR, "face_database.json")


# ─── Save face encoding to database ──────────────────────────────────
def save_face(image_path, name):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print(f"  No face found in '{image_path}'")
        return False

    db = load_database()
    db[name] = {
        "encoding":   encodings[0].tolist(),
        "image_path": image_path
    }

    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

    print(f"  Saved '{name}' -> {DB_FILE}")
    return True


# ─── Load database ────────────────────────────────────────────────────
def load_database():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)


# ─── Compare new face with stored ones ───────────────────────────────
def compare_face(new_image_path, tolerance=0.5):
    db = load_database()

    if not db:
        print("Database is empty. Please save some faces first.")
        return []

    new_image     = face_recognition.load_image_file(new_image_path)
    new_encodings = face_recognition.face_encodings(new_image)

    if not new_encodings:
        print("No face found in the new image.")
        return []

    new_encoding = new_encodings[0]
    names        = list(db.keys())
    stored       = [np.array(db[n]["encoding"]) for n in names]
    distances    = face_recognition.face_distance(stored, new_encoding)

    results = []
    for name, distance in zip(names, distances):
        confidence = round((1 - distance) * 100, 2)
        matched    = distance <= tolerance
        results.append({
            "name":       name,
            "matched":    matched,
            "confidence": confidence,
            "distance":   round(float(distance), 4)
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ─── Print comparison results ─────────────────────────────────────────
def print_results(results):
    print("\n-- Comparison Results ------------------------------------------")
    for r in results:
        status = "MATCH" if r["matched"] else "NO MATCH"
        bar    = "#" * int(r["confidence"] / 5)
        print(f"  {r['name']:<20} {status:<12} {r['confidence']:>6.2f}%  {bar}")
    print()

    matches = [r for r in results if r["matched"]]
    if matches:
        best = matches[0]
        print(f"  Best match -> {best['name']} ({best['confidence']}% confidence)")
    else:
        print("  No match found in the database.")


# ─── Annotate result on image ─────────────────────────────────────────
def annotate_result(new_image_path, results, output_path="result.png"):
    image     = face_recognition.load_image_file(new_image_path)
    locations = face_recognition.face_locations(image)

    pil_image = Image.fromarray(image)
    draw      = ImageDraw.Draw(pil_image)

    matches = [r for r in results if r["matched"]]
    label   = matches[0]["name"] + f" ({matches[0]['confidence']}%)" if matches else "Unknown"
    color   = "lime" if matches else "red"

    for (top, right, bottom, left) in locations:
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.rectangle([left, bottom, right, bottom + 24], fill=color)
        draw.text((left + 6, bottom + 4), label, fill="black")

    output_path = os.path.join(BASE_DIR, output_path)
    pil_image.save(output_path)
    print(f"  Annotated image saved -> {output_path}")
    pil_image.show()


# ─── Show all stored faces ────────────────────────────────────────────
def show_database():
    db = load_database()
    if not db:
        print("Database is empty.")
        return
    print("\n-- Stored Faces ------------------------------------------------")
    for i, name in enumerate(db, 1):
        print(f"  {i}. {name}  ({db[name]['image_path']})")
    print()


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Saving faces to database...")
    save_face(r"D:\face_feature_extract\features\person1.jpg",  "Alice")
    save_face(r"D:\face_feature_extract\features\person2.jpg",  "Bob")
    save_face(r"D:\face_feature_extract\features\person3.jpeg", "Kavi")

    show_database()

    #save the new faces below:

    print("Comparing new image...")
    results = compare_face(r"D:\face_feature_extract\features\new_person.jpeg", tolerance=0.5)

    print_results(results)
    annotate_result(r"D:\face_feature_extract\features\new_person.jpeg", results)