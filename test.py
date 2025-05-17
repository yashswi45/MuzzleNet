import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd

# === Paths ===
model_path = "final_muzzle_identifier_model.h5"
test_image_path = r"D:\HACKATHON PROJECTS D-DRIVE\Edit Muzzle\test_images\Screenshot 2025-05-10 215454.png"  # 🔁 Replace this with your image path
test_dir = "dataset/test"            # Needed to read class labels from test generator

# === Load the model ===
model = load_model(model_path)

# === Get class labels from test directory structure ===
class_labels = sorted(os.listdir(test_dir))  # Ensure consistent order

# === Load and preprocess image ===
img = load_img(test_image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# === Predict ===
predictions = model.predict(img_array)[0]  # Shape: (num_classes,)
predicted_class_index = np.argmax(predictions)
predicted_class = class_labels[predicted_class_index]
confidence = predictions[predicted_class_index]

# === Display Prediction ===
print(f"\n✅ Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")

# === Display full probability table ===
prob_df = pd.DataFrame({
    "Class": class_labels[:len(predictions)],  # Match lengths
    "Probability": predictions.tolist()
}).sort_values(by="Probability", ascending=False)

print("\n🔍 Top Predictions:")
print(prob_df.head(5))

# === Optional: Plot top 5 class probabilities ===
plt.figure(figsize=(10, 5))
plt.barh(prob_df["Class"].head(5), prob_df["Probability"].head(5), color='skyblue')
plt.gca().invert_yaxis()
plt.title("Top 5 Predicted Classes")
plt.xlabel("Probability")
plt.tight_layout()
plt.savefig("top_5_predictions.png")
plt.show()
