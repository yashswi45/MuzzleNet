import os
import shutil

# Config
AUGMENTED_SOURCE = "BeefCattle_Muzzle_Individualized"  # Where augmented images currently are
PREPROCESSED_DIR = "Preprocessed_Images"  # Where they should go

for cattle_dir in os.listdir(AUGMENTED_SOURCE):
    aug_dir = os.path.join(AUGMENTED_SOURCE, cattle_dir, "augmented")
    target_dir = os.path.join(PREPROCESSED_DIR, cattle_dir)

    if os.path.exists(aug_dir):
        os.makedirs(target_dir, exist_ok=True)
        for img in os.listdir(aug_dir):
            src = os.path.join(aug_dir, img)
            dst = os.path.join(target_dir, img)
            shutil.move(src, dst)
        print(f"Moved {len(os.listdir(aug_dir))} images to {target_dir}")