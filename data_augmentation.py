import os
import cv2
import albumentations as A
from tqdm import tqdm

PREPROCESSED_DIR = "Preprocessed_Images"
AUGMENTATIONS_PER_IMAGE = 3  # Conservative number to avoid overfitting

# Auto-clean previous augmentations
for root, _, files in os.walk(PREPROCESSED_DIR):
    for file in files:
        if "_aug" in file:
            os.remove(os.path.join(root, file))

augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
])

if __name__ == "__main__":
    for cattle_id in tqdm(os.listdir(PREPROCESSED_DIR), desc="Augmenting"):
        cattle_dir = os.path.join(PREPROCESSED_DIR, cattle_id)
        original_images = [f for f in os.listdir(cattle_dir) if "_aug" not in f]

        for img_name in original_images:
            img = cv2.imread(os.path.join(cattle_dir, img_name), cv2.IMREAD_GRAYSCALE)
            for i in range(AUGMENTATIONS_PER_IMAGE):
                augmented = augmenter(image=img)["image"]
                cv2.imwrite(
                    os.path.join(cattle_dir, f"{os.path.splitext(img_name)[0]}_aug{i}.png"),
                    augmented,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
                )
    print("✅ Augmentation complete. Old augmentations auto-cleaned.")