import cv2
import numpy as np
import pickle

from skimage.feature import local_binary_pattern


def predict_muzzle(image_path):
    # 1. Load models
    with open("D:\HACKATHON PROJECTS D-DRIVE\Edit Muzzle\Saved_Models\svm_model.pkl", "rb") as f:
        model, pca, le = pickle.load(f)

    # 2. Preprocess (same as training)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    processed = clahe.apply(gray)

    # 3. Extract features (same as training)
    orb = cv2.ORB_create(nfeatures=500)
    kp, desc = orb.detectAndCompute(processed, None)
    orb_feat = desc.mean(axis=0)[:32] if desc is not None else np.zeros(32)
    lbp = local_binary_pattern(processed, P=8, R=1, method='uniform')
    lbp_hist = np.histogram(lbp, bins=10, range=(0, 10))[0]
    features = np.concatenate([orb_feat, lbp_hist])

    # 4. Predict
    features_pca = pca.transform([features])
    pred = model.predict(features_pca)
    return le.inverse_transform(pred)[0]


# Example usage
print(predict_muzzle("D:\HACKATHON PROJECTS D-DRIVE\Edit Muzzle\test_images\WhatsApp Image 2025-05-02 at 11.56.20_f115579d.jpg"))
