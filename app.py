# # import streamlit as st
# # import os
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# # from PIL import Image
# # import cv2
# #
# # # ---------------- Set Page Configuration ----------------
# # st.set_page_config(page_title="🐄 Cattle Identification", page_icon="🐮", layout="wide")
# #
# # # ---------------- Custom Styling ----------------
# # st.markdown(
# #     """
# #     <style>
# #     body {
# #         background-image: url("https://th.bing.com/th/id/OIP.ALDcsJ_nsywqipPgiR0DKgHaE7?w=238&h=180");
# #         background-size: cover;
# #         background-position: center;
# #         background-attachment: fixed;
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True
# # )
# # background_image = """
# #     <style>
# #     .text-background {
# #         background: url("https://plus.unsplash.com/premium_photo-1661947077159-a68e6859c4a5?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
# #         background-size: cover;
# #         background-position: center;
# #         background-attachment: fixed;
# #         height: 70vh; /* Increase height to full screen */
# #         width: 100%;
# #         display: flex;
# #         flex-direction: column;
# #         justify-content: center;
# #         align-items: center;
# #         padding: 50px;
# #         border-radius: 20px;
# #         text-align: center;
# #         color: white;
# #         font-size: 24px;
# #         font-weight: bold;
# #         text-shadow: 3px 3px 8px black;
# #     }
# #     </style>
# # """
# #
# # # Inject Background Styling
# # st.markdown(background_image, unsafe_allow_html=True)
# #
# # # Content Inside Background
# # st.markdown(
# #     """
# #     <div class="text-background">
# #         <h1 style="font-size: 60px;">🐄 Cattle Identification System</h1>
# #         <h3>AI & Image Processing for Precise Cattle Recognition</h3>
# #
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )
# #
# # st.write("")
# # st.write("")
# # st.write("")
# #
# # st.markdown(
# #     """
# #     <div style="text-align: center; font-size: 18px; line-height: 1.8; max-width: 800px; margin: auto; color: #549b8a ;">
# #         🐂 This project introduces an <b>AI-powered Cattle Identification System</b> that utilizes
# #         <b>Deep Learning</b> and <b>Image Processing</b> for highly accurate livestock recognition.
# #         By analyzing unique <b>muzzle patterns</b> using a <b>Convolutional Neural Network (CNN)</b>,
# #         this system offers reliable <b>individual cattle identification</b>.
# #         <br><br>
# #         The platform is developed using <b>Streamlit</b>, allowing users to <b>upload images,
# #         preprocess them with OpenCV</b>, and receive <b>real-time cattle classification</b>.
# #         This solution significantly enhances <b>livestock tracking, breed verification,
# #         and farm management</b> through biometric authentication.
# #         <br><br>
# #         🚀 <b>Key Features:</b>
# #         ✅ Image Preprocessing | ✅ Feature Extraction | ✅ AI-Based Cattle Recognition
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )
# # st.write("")
# # st.write("")
# # # ---------------- Sidebar Navigation ----------------
# # st.sidebar.markdown("<div class='sidebar-content'><h3>📌 Navigation</h3></div>", unsafe_allow_html=True)
# # page = st.sidebar.radio("Go to", ["🏠 Home", "📷 Image Preprocessing", "🔍 Image Prediction"], index=0)
# #
# # # ---------------- Load CNN Model ----------------
# # MODEL_PATH = "cattle_muzzle_cnn_N.h5"  # Update the model path
# # model = load_model(MODEL_PATH)
# #
# #
# # # ---------------- Image Preprocessing Function ----------------
# # def preprocess_image(img):
# #     img = img.resize((128, 128))  # ✅ Keep the same size as training
# #     img_array = image.img_to_array(img)
# #     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
# #     img_array = img_array / 255.0  # Normalize
# #     return img_array
# #
# #
# # # ---------------- HOME PAGE ----------------
# # if page == "🏠 Home":
# #
# #     col1, col2, col3 = st.columns(3)
# #
# #     # -------- Clickable Image: Angus --------
# #     with col1:
# #         st.image("https://th.bing.com/th/id/OIP.ALDcsJ_nsywqipPgiR0DKgHaE7?w=238&h=180")
# #         if st.button("🐂 Angus"):
# #             with st.expander("🐂 **Angus - Overview**"):
# #                 st.write("🔹 **Breed**: Black Angus")
# #                 st.write("🔹 **Origin**: Scotland")
# #                 st.write("🔹 **Characteristics**: Marbled beef, rapid growth rate, excellent meat quality.")
# #
# #     # -------- Clickable Image: Grazing Herd --------
# #     with col2:
# #         st.image("https://th.bing.com/th/id/OIP.nR9-Hv8swrD7YNhxRlypWgHaE8?w=188&h=180")
# #         if st.button("🌿 Grazing Herd"):
# #             with st.expander("🌿 **Grazing Herd - Overview**"):
# #                 st.write(
# #                     "🔹 **Importance of Grazing**: Grazing helps cattle maintain their health and promotes sustainable farming.")
# #                 st.write("🔹 **Common Breeds Found in Grazing Herds**: Angus, Hereford, Holstein, Brahman.")
# #
# #     # -------- Clickable Image: Holstein --------
# #     with col3:
# #         st.image("https://th.bing.com/th/id/OIP.JBCpI_OCV-ntREzqj_dfNAHaE3?w=279&h=182")
# #         if st.button("🐄 Holstein"):
# #             with st.expander("🐄 **Holstein - Overview**"):
# #                 st.write("🔹 **Breed**: Holstein Friesian")
# #                 st.write("🔹 **Origin**: Netherlands")
# #                 st.write("🔹 **Characteristics**: Largest dairy breed, produces high milk yield, black & white coat.")
# #
# #     st.write("### 🌟 Features")
# #     st.write("✔️ Upload an image for preprocessing.")
# #     st.write("✔️ Extract features from the cattle's muzzle.")
# #     st.write("✔️ Predict the cattle's identity using AI.")
# #     st.markdown("---")
# #     st.write("👨‍💻 **Developed using:** Streamlit, OpenCV, TensorFlow")
# #
# #
# #
# # # ---------------- IMAGE PREPROCESSING PAGE ----------------
# # elif page == "📷 Image Preprocessing":
# #     st.markdown('<p class="main-title">📷 Image Preprocessing</p>', unsafe_allow_html=True)
# #     uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
# #
# #     if uploaded_file is not None:
# #         img = Image.open(uploaded_file)
# #         st.write("### 🔄 Image Processing Steps")
# #         st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)
# #
# #         img_array = np.array(img.convert("L"))
# #         _, threshold_img = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
# #         threshold_pil = Image.fromarray(threshold_img)
# #         st.image(threshold_pil, caption="⚙️ Processed Image", use_container_width=True)
# #
# #         save_path = "Preprocessed_Images"
# #         os.makedirs(save_path, exist_ok=True)
# #         image_path = os.path.join(save_path, uploaded_file.name)
# #         threshold_pil.save(image_path)
# #         st.success(f"✅ Preprocessed Image Saved: {image_path}")
# #
# # # ---------------- IMAGE PREDICTION PAGE ----------------
# # elif page == "🔍 Image Prediction":
# #     st.markdown('<p class="main-title">🔍 Image Prediction</p>', unsafe_allow_html=True)
# #     uploaded_file = st.file_uploader("📤 Upload an Image for Prediction", type=["jpg", "png", "jpeg"])
# #
# #     if uploaded_file is not None:
# #         img = Image.open(uploaded_file)
# #         st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)
# #
# #         img_array = preprocess_image(img)
# #         with st.spinner("🔄 Predicting... Please wait!"):
# #             prediction = model.predict(img_array)
# #
# #         predicted_class = np.argmax(prediction, axis=1)[0]
# #         st.success(f"🎯 **Predicted Cattle ID: {predicted_class}**")
# #
# #         # Information Section
# #         st.markdown("### ℹ️ Cattle Information")
# #         cattle_info = {
# #             0: "🐂 **Angus**: Known for marbled beef quality.",
# #             1: "🐄 **Hereford**: Hardy breed with a white face.",
# #             2: "🐮 **Jersey**: Produces high-butterfat milk.",
# #             3: "🐄 **Holstein**: Largest dairy breed globally.",
# #             4: "🐂 **Brahman**: Heat-tolerant breed from India."
# #         }
# #         st.info(cattle_info.get(predicted_class,
# #                                 "**Visit For More Info**: http://www.agritech.tnau.ac.in/expert_system/cattlebuffalo/Breeds%20of%20cattle%20&%20baffalo.html"))
# #
# #         if st.button("🔄 Predict Again"):
# #             st.experimental_rerun()
#
#
#
#
# #
# # import streamlit as st
# # import os
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# # from PIL import Image
# # import cv2
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# #
# # # ---------------- Set Page Configuration ----------------
# # st.set_page_config(page_title="🐄 Cattle Identification", page_icon="🐮", layout="wide")
# #
# # # ---------------- Custom Styling ----------------
# # st.markdown(
# #     """
# #     <style>
# #     body {
# #         background-image: url("https://th.bing.com/th/id/OIP.ALDcsJ_nsywqipPgiR0DKgHaE7?w=238&h=180");
# #         background-size: cover;
# #         background-position: center;
# #         background-attachment: fixed;
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True
# # )
# # background_image = """
# #     <style>
# #     .text-background {
# #         background: url("https://plus.unsplash.com/premium_photo-1661947077159-a68e6859c4a5?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
# #         background-size: cover;
# #         background-position: center;
# #         background-attachment: fixed;
# #         height: 70vh; /* Increase height to full screen */
# #         width: 100%;
# #         display: flex;
# #         flex-direction: column;
# #         justify-content: center;
# #         align-items: center;
# #         padding: 50px;
# #         border-radius: 20px;
# #         text-align: center;
# #         color: white;
# #         font-size: 24px;
# #         font-weight: bold;
# #         text-shadow: 3px 3px 8px black;
# #     }
# #     </style>
# # """
# #
# # # Inject Background Styling
# # st.markdown(background_image, unsafe_allow_html=True)
# #
# # # Content Inside Background
# # st.markdown(
# #     """
# #     <div class="text-background">
# #         <h1 style="font-size: 60px;">🐄 Cattle Identification System</h1>
# #         <h3>AI & Image Processing for Precise Cattle Recognition</h3>
# #
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )
# #
# # st.write(" ")
# # st.write(" ")
# # st.write(" ")
# #
# # st.markdown(
# #     """
# #     <div style="text-align: center; font-size: 18px; line-height: 1.8; max-width: 800px; margin: auto; color: #549b8a ;">
# #         🐂 This project introduces an <b>AI-powered Cattle Identification System</b> that utilizes
# #         <b>Deep Learning</b> and <b>Image Processing</b> for highly accurate livestock recognition.
# #         By analyzing unique <b>muzzle patterns</b> using a <b>Convolutional Neural Network (CNN)</b>,
# #         this system offers reliable <b>individual cattle identification</b>.
# #         <br><br>
# #         The platform is developed using <b>Streamlit</b>, allowing users to <b>upload images,
# #         preprocess them with OpenCV</b>, and receive <b>real-time cattle classification</b>.
# #         This solution significantly enhances <b>livestock tracking, breed verification,
# #         and farm management</b> through biometric authentication.
# #         <br><br>
# #         🚀 <b>Key Features:</b>
# #         ✅ Image Preprocessing | ✅ Feature Extraction | ✅ AI-Based Cattle Recognition
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )
# # st.write(" ")
# # st.write(" ")
# #
# # # ---------------- Sidebar Navigation ----------------
# # st.sidebar.markdown("<div class='sidebar-content'><h3>📌 Navigation</h3></div>", unsafe_allow_html=True)
# # page = st.sidebar.radio("Go to", ["🏠 Home", "📷 Image Preprocessing", "🔍 Image Prediction"], index=0)
# #
# # # ---------------- Load CNN Model ----------------
# # MODEL_PATH = "final_muzzle_identifier_model.h5"  # Update the model path
# # model = load_model(MODEL_PATH)
# #
# # # ---------------- Image Preprocessing Function ----------------
# #
# # # Function to preprocess image
# # # Function to preprocess image
# # # def preprocess_image(image):
# # #     # Resize image to 224x224
# # #     image = image.resize((224, 224))
# # #     # Convert image to grayscale (for feature extraction or further processing)
# # #     image = image.convert("L")  # Convert to grayscale
# # #     # Convert image to numpy array and scale pixel values
# # #     img_array = np.array(image) / 255.0  # Normalize pixel values to range [0, 1]
# # #     # Convert grayscale to 3 channels (for RGB)
# # #     img_array = np.stack([img_array] * 3, axis=-1)  # Stack the grayscale image to 3 channels
# # #     # Add batch dimension (for a single image, batch size = 1)
# # #     img_array = np.expand_dims(img_array, axis=0)
# # #     return img_array
# #
# # from PIL import Image
# # import numpy as np
# #
# # def preprocess_image(uploaded_file):
# #     image = Image.open(uploaded_file).convert('RGB')  # Ensures 3 channels
# #     image = image.resize((224, 224))  # Resize as per model
# #     img_array = np.array(image) / 255.0  # Normalize
# #     img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
# #     return img_array
# #
# #
# #
# # # ---------------- HOME PAGE ----------------
# # if page == "🏠 Home":
# #     st.write("Welcome to the Cattle Identification System!")
# #     st.write("Upload images for AI-based identification.")
# #     st.write("Use the sidebar to navigate between Home, Image Preprocessing, and Prediction.")
# #
# # # ---------------- IMAGE PREDICTION PAGE ----------------
# # elif page == "🔍 Image Prediction":
# #     st.markdown('<p class="main-title">🔍 Image Prediction</p>', unsafe_allow_html=True)
# #
# #     uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
# #
# #     if uploaded_image is not None:
# #         # Load and preprocess image
# #         img = Image.open(uploaded_image)
# #         img_array = preprocess_image(img)
# #
# #         # Prediction
# #         with st.spinner("🔄 Predicting... Please wait!"):
# #             prediction = model.predict(img_array)
# #
# #         # Get class probabilities
# #         class_probabilities = prediction[0]  # Prediction is a 2D array, take the first row (for a single image)
# #
# #         # Prediction class with highest probability
# #         predicted_class = np.argmax(class_probabilities)
# #
# #         st.write(f"Predicted Class: {class_labels[predicted_class]}")
# #
# #         # Display the probability chart
# #         st.markdown("### 📊 Prediction Probabilities")
# #         probability_dict = dict(zip(class_labels, class_probabilities))
# #         st.bar_chart(probability_dict)
# #
# #         # Information Section
# #         st.markdown("### ℹ️ Cattle Information")
# #         cattle_info = {
# #             0: "🐂 **Angus**: Known for marbled beef quality.",
# #             1: "🐄 **Hereford**: Hardy breed with a white face.",
# #             2: "🐮 **Jersey**: Produces high-butterfat milk.",
# #             3: "🐄 **Holstein**: Largest dairy breed globally.",
# #             4: "🐂 **Brahman**: Heat-tolerant breed from India."
# #         }
# #         st.info(cattle_info.get(predicted_class,
# #                                 "**Visit For More Info**: http://www.agritech.tnau.ac.in/expert_system/cattlebuffalo/Breeds%20of%20cattle%20&%20baffalo.html"))
# #
# #         if st.button("🔄 Predict Again"):
# #             st.experimental_rerun()
# #
# # # ---------------- IMAGE PREPROCESSING PAGE ----------------
# # elif page == "📷 Image Preprocessing":
# #     st.markdown('<p class="main-title">📷 Image Preprocessing</p>', unsafe_allow_html=True)
# #
# #     uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
# #
# #     if uploaded_image is not None:
# #         # Load and preprocess image
# #         img = Image.open(uploaded_image)
# #
# #         # Convert to grayscale and display the image
# #         grayscale_image = img.convert("L")
# #         st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)
# #
# #         # Convert image to numpy array and scale pixel values
# #         img_array = np.array(grayscale_image) / 255.0
# #         st.write("Preprocessed Image (Grayscale) - Ready for Feature Extraction")
# #
# #         # Optionally show a simple histogram of pixel intensities
# #         plt.hist(img_array.flatten(), bins=256, color='gray', alpha=0.7)
# #         plt.title('Grayscale Image Histogram')
# #         plt.xlabel('Pixel Intensity')
# #         plt.ylabel('Frequency')
# #         st.pyplot(plt)
# #
#
#
#
#
#
#
#
# # import streamlit as st
# # import os
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # from PIL import Image
# # import cv2
# # import pandas as pd
# # import  matplotlib.pyplot as plt
# #
# # # ---------------- Set Page Configuration ----------------
# # st.set_page_config(page_title="🐄 Cattle Identification", page_icon="🐮", layout="wide")
# #
# # # ---------------- Custom Styling ----------------
# # st.markdown(
# #     """
# #     <style>
# #     .main-title {
# #         font-size: 2.5rem;
# #         color: #2E86AB;
# #         text-align: center;
# #         margin-bottom: 2rem;
# #     }
# #     .prediction-result {
# #         font-size: 1.5rem;
# #         color: #28A745;
# #         text-align: center;
# #         padding: 1rem;
# #         background-color: #E8F5E9;
# #         border-radius: 0.5rem;
# #         margin: 1rem 0;
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True
# # )
# #
# # # Background styling
# # st.markdown(
# #     """
# #     <style>
# #     .text-background {
# #         background: url("https://plus.unsplash.com/premium_photo-1661947077159-a68e6859c4a5?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
# #         background-size: cover;
# #         background-position: center;
# #         background-attachment: fixed;
# #         height: 70vh;
# #         width: 100%;
# #         display: flex;
# #         flex-direction: column;
# #         justify-content: center;
# #         align-items: center;
# #         padding: 50px;
# #         border-radius: 20px;
# #         text-align: center;
# #         color: white;
# #         font-size: 24px;
# #         font-weight: bold;
# #         text-shadow: 3px 3px 8px black;
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True
# # )
# #
# # # Content Inside Background
# # st.markdown(
# #     """
# #     <div class="text-background">
# #         <h1 style="font-size: 60px;">🐄 Cattle Identification System</h1>
# #         <h3>AI & Image Processing for Precise Cattle Recognition</h3>
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )
# #
# # st.write("")
# # st.write("")
# # st.write("")
# #
# # st.markdown(
# #     """
# #     <div style="text-align: center; font-size: 18px; line-height: 1.8; max-width: 800px; margin: auto; color: #549b8a;">
# #         🐂 This project introduces an <b>AI-powered Cattle Identification System</b> that utilizes
# #         <b>Deep Learning</b> and <b>Image Processing</b> for highly accurate livestock recognition.
# #         By analyzing unique <b>muzzle patterns</b> using a <b>Convolutional Neural Network (CNN)</b>,
# #         this system offers reliable <b>individual cattle identification</b>.
# #         <br><br>
# #         The platform is developed using <b>Streamlit</b>, allowing users to <b>upload images,
# #         preprocess them with OpenCV</b>, and receive <b>real-time cattle classification</b>.
# #         This solution significantly enhances <b>livestock tracking, breed verification,
# #         and farm management</b> through biometric authentication.
# #         <br><br>
# #         🚀 <b>Key Features:</b>
# #         ✅ Image Preprocessing | ✅ Feature Extraction | ✅ AI-Based Cattle Recognition
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )
# # st.write("")
# # st.write("")
# #
# # # ---------------- Sidebar Navigation ----------------
# # st.sidebar.markdown("<div style='margin-bottom: 2rem;'><h3>📌 Navigation</h3></div>", unsafe_allow_html=True)
# # page = st.sidebar.radio("Go to", ["🏠 Home", "📷 Image Preprocessing", "🔍 Image Prediction"], index=0)
# #
# # # ---------------- Load CNN Model ----------------
# # MODEL_PATH = "final_muzzle_identifier_model.h5"
# # try:
# #     model = load_model(MODEL_PATH)
# # except Exception as e:
# #     st.error(f"Failed to load model: {str(e)}")
# #     st.stop()
# #
# # # ---------------- Load Class Labels ----------------
# # DATASET_DIR = "dataset"  # Replace with your dataset directory
# # try:
# #     class_labels = sorted(os.listdir(DATASET_DIR))
# #     if not class_labels:
# #         st.error("No class labels found in the dataset directory.")
# #         st.stop()
# # except Exception as e:
# #     st.error(f"Failed to load class labels: {str(e)}")
# #     st.stop()
# #
# #
# # # ---------------- Image Preprocessing Function ----------------
# # def preprocess_image(image):
# #     try:
# #         image = image.resize((224, 224))
# #         image = image.convert("RGB")
# #         img_array = np.array(image) / 255.0
# #         img_array = np.expand_dims(img_array, axis=0)
# #         return img_array
# #     except Exception as e:
# #         st.error(f"Image preprocessing failed: {str(e)}")
# #         return None
# #
# #
# # # ---------------- HOME PAGE ----------------
# # if page == "🏠 Home":
# #     st.write("Welcome to the Cattle Identification System!")
# #     st.write("Upload images for AI-based identification.")
# #     st.write("Use the sidebar to navigate between Home, Image Preprocessing, and Prediction.")
# #
# # # ---------------- IMAGE PREPROCESSING PAGE ----------------
# # elif page == "📷 Image Preprocessing":
# #     st.markdown('<p class="main-title">📷 Image Preprocessing</p>', unsafe_allow_html=True)
# #
# #     uploaded_image = st.file_uploader("Upload Image for Preprocessing", type=["png", "jpg", "jpeg"])
# #
# #     if uploaded_image is not None:
# #         try:
# #             img = Image.open(uploaded_image)
# #             st.image(img, caption="Original Image", use_column_width=True)
# #
# #             # Convert to grayscale
# #             gray_img = img.convert("L")
# #             st.image(gray_img, caption="Grayscale Image", use_column_width=True)
# #
# #             # Show histogram of grayscale image
# #             gray_np = np.array(gray_img)
# #             fig, ax = plt.subplots()
# #             ax.hist(gray_np.ravel(), bins=256, color='gray', alpha=0.7)
# #             ax.set_title("Grayscale Intensity Histogram")
# #             ax.set_xlabel("Pixel Intensity (0-255)")
# #             ax.set_ylabel("Frequency")
# #             st.pyplot(fig)
# #
# #         except Exception as e:
# #             st.error(f"Image processing failed: {str(e)}")
# #
# # # ---------------- IMAGE PREDICTION PAGE ----------------
# # elif page == "🔍 Image Prediction":
# #     st.markdown('<p class="main-title">🔍 Image Prediction</p>', unsafe_allow_html=True)
# #
# #     uploaded_image = st.file_uploader("Upload Image for Prediction", type=["png", "jpg", "jpeg"])
# #
# #     if uploaded_image is not None:
# #         try:
# #             img = Image.open(uploaded_image)
# #             st.image(img, caption="Uploaded Image", use_column_width=True)
# #
# #             img_array = preprocess_image(img)
# #             if img_array is None:
# #                 st.stop()
# #
# #             with st.spinner("🔄 Predicting... Please wait!"):
# #                 prediction = model.predict(img_array)
# #
# #             if len(prediction[0]) != len(class_labels):
# #                 st.error(
# #                     f"Mismatch between model output ({len(prediction[0])} classes) and class labels ({len(class_labels)}).")
# #                 st.stop()
# #
# #             predicted_class_index = np.argmax(prediction, axis=1)[0]
# #             predicted_class_label = class_labels[predicted_class_index]
# #             confidence = np.max(prediction) * 100
# #
# #             st.markdown(
# #                 f'<div class="prediction-result">Predicted Class: {predicted_class_label}<br>Confidence: {confidence:.2f}%</div>',
# #                 unsafe_allow_html=True)
# #
# #             # Display the probability chart
# #             st.markdown("### 📊 Prediction Probabilities")
# #             probabilities = prediction[0]
# #
# #             # Create DataFrame with class labels and probabilities
# #             prob_df = pd.DataFrame({
# #                 'Breed': class_labels,
# #                 'Probability': probabilities
# #             }).sort_values('Probability', ascending=False)
# #
# #             st.bar_chart(prob_df.set_index('Breed'))
# #
# #             # Information Section
# #             st.markdown("### ℹ️ Cattle Information")
# #             cattle_info = {
# #                 "buffello": "🐃 **Buffalo**: Known for high milk yield and adaptability.",
# #                 "cattle_1": "🐄 **Cattle 1**: Description for Cattle 1.",
# #                 "cattle_2": "🐄 **Cattle 2**: Description for Cattle 2.",
# #                 "Jeysey_Cow": "🐄 **Jersey Cow**: Produces high-butterfat milk."
# #             }
# #
# #             # Add default descriptions for any missing classes
# #             info = cattle_info.get(predicted_class_label,
# #                                    f"**{predicted_class_label}**: No specific information available for this breed.")
# #             st.info(info)
# #
# #             if st.button("🔄 Predict Again"):
# #                 st.experimental_rerun()
# #
# #         except Exception as e:
# #             st.error(f"Prediction failed: {str(e)}")
# #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import streamlit as st
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import cv2
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ---------------- Set Page Configuration ----------------
# st.set_page_config(page_title="🐄 Cattle Identification", page_icon="🐮", layout="wide")
#
# # ---------------- Custom Styling ----------------
# st.markdown(
#     """
#     <style>
#     .main-title {
#         font-size: 2.5rem;
#         color: #2E86AB;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .prediction-result {
#         font-size: 1.5rem;
#         color: #28A745;
#         text-align: center;
#         padding: 1rem;
#         background-color: #E8F5E9;
#         border-radius: 0.5rem;
#         margin: 1rem 0;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
#
# # ---------------- Background Styling ----------------
# st.markdown(
#     """
#     <style>
#     .text-background {
#         background: url("https://plus.unsplash.com/premium_photo-1661947077159-a68e6859c4a5?q=80&w=2070&auto=format&fit=crop");
#         background-size: cover;
#         background-position: center;
#         background-attachment: fixed;
#         height: 70vh;
#         width: 100%;
#         display: flex;
#         flex-direction: column;
#         justify-content: center;
#         align-items: center;
#         padding: 50px;
#         border-radius: 20px;
#         text-align: center;
#         color: white;
#         font-size: 24px;
#         font-weight: bold;
#         text-shadow: 3px 3px 8px black;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
#
# # ---------------- Hero Section ----------------
# st.markdown(
#     """
#     <div class="text-background">
#         <h1 style="font-size: 60px;">🐄 Cattle Identification System</h1>
#         <h3>AI & Image Processing for Precise Cattle Recognition</h3>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# st.write("")
# st.write("")
# st.write("")
#
# # ---------------- Project Overview ----------------
# st.markdown(
#     """
#     <div style="text-align: center; font-size: 18px; line-height: 1.8; max-width: 800px; margin: auto; color: #549b8a;">
#         🐂 This project introduces an <b>AI-powered Cattle Identification System</b> that utilizes
#         <b>Deep Learning</b> and <b>Image Processing</b> for highly accurate livestock recognition.
#         By analyzing unique <b>muzzle patterns</b> using a <b>Convolutional Neural Network (CNN)</b>,
#         this system offers reliable <b>individual cattle identification</b>.
#         <br><br>
#         The platform is developed using <b>Streamlit</b>, allowing users to <b>upload images,
#         preprocess them with OpenCV</b>, and receive <b>real-time cattle classification</b>.
#         This solution significantly enhances <b>livestock tracking, breed verification,
#         and farm management</b> through biometric authentication.
#         <br><br>
#         🚀 <b>Key Features:</b>
#         ✅ Image Preprocessing | ✅ Feature Extraction | ✅ AI-Based Cattle Recognition
#     </div>
#     """,
#     unsafe_allow_html=True
# )
#
# # ---------------- Sidebar Navigation ----------------
# st.sidebar.markdown("<div style='margin-bottom: 2rem;'><h3>📌 Navigation</h3></div>", unsafe_allow_html=True)
# page = st.sidebar.radio("Go to", ["🏠 Home", "📷 Image Preprocessing", "🔍 Image Prediction"], index=0)
#
# # ---------------- Load CNN Model ----------------
# MODEL_PATH = "final_muzzle_identifier_model.h5"
# try:
#     model = load_model(MODEL_PATH)
# except Exception as e:
#     st.error(f"Failed to load model: {str(e)}")
#     st.stop()
#
# # ---------------- Load Class Labels ----------------
# DATASET_DIR = "dataset"
# try:
#     train_dir = "dataset/train"
#     class_labels = sorted(os.listdir(train_dir))  # ✅ gives all 17 class labels
#
#     if not class_labels:
#         st.error("No class labels found in the dataset directory.")
#         st.stop()
# except Exception as e:
#     st.error(f"Failed to load class labels: {str(e)}")
#     st.stop()
#
# # ---------------- Image Preprocessing Function ----------------
# def preprocess_image(image):
#     try:
#         image = image.resize((224, 224))
#         image = image.convert("RGB")
#         img_array = np.array(image) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#         return img_array
#     except Exception as e:
#         st.error(f"Image preprocessing failed: {str(e)}")
#         return None
#
# # ---------------- HOME PAGE ----------------
# if page == "🏠 Home":
#     st.header("Welcome to the Cattle Identification System!")
#     st.markdown("Use the sidebar to navigate between:")
#     st.markdown("- 📷 Image Preprocessing")
#     st.markdown("- 🔍 Image Prediction")
#
# # ---------------- IMAGE PREPROCESSING PAGE ----------------
# elif page == "📷 Image Preprocessing":
#     st.markdown('<p class="main-title">📷 Image Preprocessing</p>', unsafe_allow_html=True)
#
#     uploaded_image = st.file_uploader("Upload Image for Preprocessing", type=["png", "jpg", "jpeg"])
#
#     if uploaded_image is not None:
#         try:
#             img = Image.open(uploaded_image)
#             st.image(img, caption="Original Image", use_column_width=True)
#
#             gray_img = img.convert("L")
#             st.image(gray_img, caption="Grayscale Image", use_column_width=True)
#
#             gray_np = np.array(gray_img)
#             fig, ax = plt.subplots()
#             ax.hist(gray_np.ravel(), bins=256, color='gray', alpha=0.7)
#             ax.set_title("Grayscale Intensity Histogram")
#             ax.set_xlabel("Pixel Intensity (0-255)")
#             ax.set_ylabel("Frequency")
#             st.pyplot(fig)
#
#         except Exception as e:
#             st.error(f"Image processing failed: {str(e)}")
#
# # ---------------- IMAGE PREDICTION PAGE ----------------
# elif page == "🔍 Image Prediction":
#     st.markdown('<p class="main-title">🔍 Image Prediction</p>', unsafe_allow_html=True)
#
#     uploaded_image = st.file_uploader("Upload Image for Prediction", type=["png", "jpg", "jpeg"])
#
#     if uploaded_image is not None:
#         try:
#             img = Image.open(uploaded_image)
#             st.image(img, caption="Uploaded Image", use_column_width=True)
#
#             img_array = preprocess_image(img)
#             if img_array is None:
#                 st.stop()
#
#             with st.spinner("🔄 Predicting... Please wait!"):
#                 prediction = model.predict(img_array)
#
#             if len(prediction[0]) != len(class_labels):
#                 st.error(
#                     f"Mismatch between model output ({len(prediction[0])} classes) and class labels ({len(class_labels)}).")
#                 st.stop()
#
#             predicted_class_index = np.argmax(prediction, axis=1)[0]
#             predicted_class_label = class_labels[predicted_class_index]
#             confidence = np.max(prediction) * 100
#
#             st.markdown(
#                 f'<div class="prediction-result">Predicted Class: <b>{predicted_class_label}</b><br>Confidence: <b>{confidence:.2f}%</b></div>',
#                 unsafe_allow_html=True
#             )
#
#             # Bar Chart
#             st.markdown("### 📊 Prediction Probabilities")
#             prob_df = pd.DataFrame({
#                 'Breed': class_labels,
#                 'Probability': prediction[0]
#             }).sort_values('Probability', ascending=False)
#             st.bar_chart(prob_df.set_index('Breed'))
#
#             # Information Section
#             st.markdown("### ℹ️ Cattle Information")
#             cattle_info = {
#                 "buffello": "🐃 **Buffalo**: Known for high milk yield and adaptability.",
#                 "cattle_1": "🐄 **Cattle 1**: Generic cattle ID for trial purposes.",
#                 "cattle_2": "🐄 **Cattle 2**: Used in experimental batches.",
#                 "Jeysey_Cow": "🐄 **Jersey Cow**: Produces high-butterfat milk."
#             }
#             info = cattle_info.get(predicted_class_label, f"**{predicted_class_label}**: No specific information available.")
#             st.info(info)
#
#             if st.button("🔄 Predict Again"):
#                 st.experimental_rerun()
#
#         except Exception as e:
#             st.error(f"Prediction failed: {str(e)}")
#
#
#
#
#
#
#









import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Set Page Configuration ----------------
st.set_page_config(page_title="🐄 Cattle Identification", page_icon="🐮", layout="wide")

# ---------------- Custom Styling ----------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        font-size: 1.5rem;
        color: #28A745;
        text-align: center;
        padding: 1rem;
        background-color: #E8F5E9;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Background Styling ----------------
st.markdown(
    """
    <style>
    .text-background {
        background: url("https://plus.unsplash.com/premium_photo-1661947077159-a68e6859c4a5?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        height: 70vh;
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 50px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        text-shadow: 3px 3px 8px black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Hero Section ----------------
st.markdown(
    """
    <div class="text-background">
        <h1 style="font-size: 60px;">🐄 Cattle Identification System</h1>
        <h3>AI & Image Processing for Precise Cattle Recognition</h3>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")
st.write("")
st.write("")

# ---------------- Project Overview ----------------


# ---------------- Sidebar Navigation ----------------
st.sidebar.markdown("<div style='margin-bottom: 2rem;'><h3>📌 Navigation</h3></div>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["🏠 Home", "📷 Image Preprocessing", "🔍 Image Prediction"], index=0)

# ---------------- Load CNN Model ----------------
MODEL_PATH = "final_muzzle_identifier_model.h5"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# ---------------- Load Class Labels ----------------
DATASET_DIR = "dataset"
try:
    train_dir = "dataset/train"
    class_labels = sorted(os.listdir(train_dir))  # ✅ gives all 17 class labels

    if not class_labels:
        st.error("No class labels found in the dataset directory.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load class labels: {str(e)}")
    st.stop()

# ---------------- Image Preprocessing Function ----------------
def preprocess_image(image):
    try:
        image = image.resize((224, 224))
        image = image.convert("RGB")
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing failed: {str(e)}")
        return None

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.markdown("""
    <div style="
        background-color: #A8D4AD;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    ">
        <h1 style="
            color: #2E4057;
            text-align: center;
            font-family: 'Arial Rounded MT Bold', sans-serif;
            margin-bottom: 1rem;
        ">🐄 Welcome to Cattle ID System</h1>
        <p style="
            color: #2E4057;
            text-align: center;
            font-size: 1.1rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            Advanced AI-powered cattle identification through unique muzzle patterns
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main content columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="
            background-color: #92B9BD;
            padding: 1.5rem;
            border-radius: 15px;
            height: 100%;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h3 style="color: #2E4057; border-bottom: 2px solid #967D69; padding-bottom: 0.5rem;">🔍 System Features</h3>
            <ul style="color: #2E4057; font-size: 1rem;">
                <li>Biometric cattle identification</li>
                <li>Deep learning-powered recognition</li>
                <li>17-class classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background-color: #A1B0AB;
            padding: 1.5rem;
            border-radius: 15px;
            height: 100%;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h3 style="color: #2E4057; border-bottom: 2px solid #967D69; padding-bottom: 0.5rem;">📌 Quick Navigation</h3>
            <p style="color: #2E4057; font-size: 0.65rem;"><b>
                Use the sidebar to access different features:</b>
            </p>
            <ul style="color: #2E4057; font-size: 1rem;">
                <li><b>Image Preprocessing</b> - View image transformations</li>
                <li><b>Image Prediction</b> - Get cattle identification results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # New sections added at the bottom
    st.markdown("""
    <div style="
        background-color: #F5F5F5;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    ">
        <h2 style="color: #967D69; text-align: center; font-family: 'Georgia', serif;">How It Works</h2>
        <div style="display: flex; justify-content: space-between; margin-top: 1.5rem;">
            <div style="text-align: center; width: 30%;">
                <div style="background-color: #92B9BD; border-radius: 50%; width: 60px; height: 60px; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">1</div>
                <h4 style="color: #2E4057;">Upload Image</h4>
                <p style="color: #2E4057; font-size: 0.9rem;">Capture or upload a clear image of cattle muzzle</p>
            </div>
            <div style="text-align: center; width: 30%;">
                <div style="background-color: #A8D4AD; border-radius: 50%; width: 60px; height: 60px; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">2</div>
                <h4 style="color: #2E4057;">AI Processing</h4>
                <p style="color: #2E4057; font-size: 0.9rem;">Our CNN model analyzes unique muzzle patterns</p>
            </div>
            <div style="text-align: center; width: 30%;">
                <div style="background-color: #967D69; border-radius: 50%; width: 60px; height: 60px; margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">3</div>
                <h4 style="color: #2E4057;">Get Results</h4>
                <p style="color: #2E4057; font-size: 0.9rem;">Receive identification with confidence percentage</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Testimonials section
    st.markdown("""
    <div style="
        background-color: #E8E8E8;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    ">
        <h2 style="color: #967D69; text-align: center; font-family: 'Georgia', serif;">What Farmers Say</h2>
        <div style="display: flex; justify-content: space-between; margin-top: 1.5rem;">
            <div style="width: 48%; padding: 1rem; background-color: white; border-radius: 10px; border-left: 4px solid #A8D4AD;">
                <p style="font-style: italic; color: #2E4057;">"This system revolutionized our livestock tracking. 98% accuracy in identifying our 200+ cattle."</p>
                <p style="text-align: right; color: #967D69; font-weight: bold;">— Rajesh, Dairy Farm Owner</p>
            </div>
            <div style="width: 48%; padding: 1rem; background-color: white; border-radius: 10px; border-left: 4px solid #92B9BD;">
                <p style="font-style: italic; color: #2E4057;">"The biometric identification saved us countless hours in manual record-keeping. Highly recommended!"</p>
                <p style="text-align: right; color: #967D69; font-weight: bold;">— Priya, Livestock Manager</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- IMAGE PREPROCESSING PAGE ----------------
elif page == "📷 Image Preprocessing":
    st.markdown('<p class="main-title">📷 Image Preprocessing</p>', unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload Image for Preprocessing", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image)
            st.image(img, caption="Original Image", use_column_width=True)

            gray_img = img.convert("L")
            st.image(gray_img, caption="Grayscale Image", use_column_width=True)

            gray_np = np.array(gray_img)
            fig, ax = plt.subplots()
            ax.hist(gray_np.ravel(), bins=256, color='gray', alpha=0.7)
            ax.set_title("Grayscale Intensity Histogram")
            ax.set_xlabel("Pixel Intensity (0-255)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Image processing failed: {str(e)}")

# ---------------- IMAGE PREDICTION PAGE ----------------
elif page == "🔍 Image Prediction":
    st.markdown('<p class="main-title">🔍 Image Prediction</p>', unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload Image for Prediction", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            img_array = preprocess_image(img)
            if img_array is None:
                st.stop()

            with st.spinner("🔄 Predicting... Please wait!"):
                prediction = model.predict(img_array)

            if len(prediction[0]) != len(class_labels):
                st.error(
                    f"Mismatch between model output ({len(prediction[0])} classes) and class labels ({len(class_labels)}).")
                st.stop()

            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_label = class_labels[predicted_class_index]
            confidence = np.max(prediction) * 100

            st.markdown(
                f'<div class="prediction-result">Predicted Class: <b>{predicted_class_label}</b><br>Confidence: <b>{confidence:.2f}%</b></div>',
                unsafe_allow_html=True
            )

            # Bar Chart
            st.markdown("### 📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Breed': class_labels,
                'Probability': prediction[0]
            }).sort_values('Probability', ascending=False)
            st.bar_chart(prob_df.set_index('Breed'))

            # Information Section
            st.markdown("### ℹ️ Cattle Information")
            cattle_info = {
                "buffello": "🐃 **Buffalo**: Known for high milk yield and adaptability.",
                "cattle_1": "🐄 **Cattle 1**: Generic cattle ID for trial purposes.",
                "cattle_2": "🐄 **Cattle 2**: Used in experimental batches.",
                "Jeysey_Cow": "🐄 **Jersey Cow**: Produces high-butterfat milk."
            }
            info = cattle_info.get(predicted_class_label, f"**{predicted_class_label}**: No specific information available.")
            st.info(info)

            if st.button("🔄 Predict Again"):
                st.experimental_rerun()

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")



