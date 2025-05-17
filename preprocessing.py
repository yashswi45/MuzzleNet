
import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

INPUT_DIR = "BeefCattle_Muzzle_database/BeefCattle_Muzzle_Individualized"
OUTPUT_DIR = "Preprocessed_Images"


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
    return cv2.bilateralFilter(denoised, 9, 75, 75)


if __name__ == "__main__":
    # Auto-clean output directory
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)

    for cattle_id in tqdm(os.listdir(INPUT_DIR), desc="Preprocessing"):
        cattle_path = os.path.join(INPUT_DIR, cattle_id)
        if not os.path.isdir(cattle_path):
            continue

        os.makedirs(os.path.join(OUTPUT_DIR, cattle_id), exist_ok=True)

        for img_name in os.listdir(cattle_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                processed = preprocess_image(os.path.join(cattle_path, img_name))
                if processed is not None:
                    cv2.imwrite(
                        os.path.join(OUTPUT_DIR, cattle_id, f"{os.path.splitext(img_name)[0]}.png"),
                        processed,
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
                    )
    print(f"✅ Preprocessing complete. Output: {OUTPUT_DIR}")