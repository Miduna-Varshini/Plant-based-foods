import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import gdown
from gemini_utils import analyze_product

st.title("🥭 Fruit & Vegetable Analyzer")

st.write("""
Upload an image of a fruit or vegetable, and we'll identify it and provide insights using AI.
""")

# Download the model from Google Drive
url = "https://drive.google.com/uc?id=1zFL5xeBEjXfGRpHnJZjvXJjlEv7zUi_p"
output = "fruit_vegetable_model.h5"
gdown.download(url, output, quiet=False)

# Load the model
model = load_model(output)

# Define classes
fruit_classes = ['Apple', 'Banana', 'Mango', 'Orange']
veg_classes = ['Carrot', 'Broccoli', 'Spinach', 'Potato']

# File uploader
uploaded_file = st.file_uploader("Upload Fruit or Vegetable Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    if predicted_index < len(fruit_classes):
        predicted_class = fruit_classes[predicted_index]
    else:
        predicted_class = veg_classes[predicted_index - len(fruit_classes)]

    st.subheader("🧠 Detected Category:")
    st.success(predicted_class)

    # Gemini AI analysis
    st.subheader("🤖 Gemini AI Analysis")
    analysis = analyze_product(predicted_class)
    st.write(analysis)

    st.subheader("📊 Benefits and Uses")
    st.write("Detailed benefits and uses of the identified fruit or vegetable will be shown here.")
