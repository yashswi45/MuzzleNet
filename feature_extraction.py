import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

# Config
PREPROCESSED_DIR = "Preprocessed_Images"
FEATURES_DIR = "Extracted_Features"
FEATURE_METHOD = "ORB"  # ORB or SIFT

# Initialize extractor
if FEATURE_METHOD == "SIFT":
    extractor = cv2.SIFT_create()
    feature_dim = 128
else:
    extractor = cv2.ORB_create(nfeatures=500)
    feature_dim = 32


def extract_features(img_path):
    """Enhanced feature extraction with BoW"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Standardize size
    img = cv2.resize(img, (224, 224))

    # Extract features
    kp, desc = extractor.detectAndCompute(img, None)
    if desc is None:
        return np.zeros(feature_dim)

    # Use mean descriptor
    return desc.mean(axis=0)[:feature_dim]


# Process all classes
os.makedirs(FEATURES_DIR, exist_ok=True)

for cattle_id in tqdm(os.listdir(PREPROCESSED_DIR), desc="Extracting"):
    cattle_dir = os.path.join(PREPROCESSED_DIR, cattle_id)
    features = []

    for img_name in os.listdir(cattle_dir):
        feature = extract_features(os.path.join(cattle_dir, img_name))
        if feature is not None:
            features.append(feature)

    # Save features per class
    with open(os.path.join(FEATURES_DIR, f"{cattle_id}_features.pkl"), "wb") as f:
        pickle.dump(features, f)

print(f"✅ Features saved to {FEATURES_DIR}")