from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from mtcnn import MTCNN
import numpy as np
import pyperclip

# Load Models & Data
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Function to save uploaded image
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

# Function to extract features from an image
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if len(results) == 0:
        st.error("❌ No face detected! Please upload a clear image with a visible face.")
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    
    return result

# Function to find best match
def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in feature_list]
    index_pos, resemblance = sorted(enumerate(similarity), reverse=True, key=lambda x: x[1])[0]
    return index_pos, round(resemblance * 100, 2)

# Function to generate beautified comparison image
def generate_comparison_image(user_image, celeb_image, celeb_name, resemblance_percentage):
    user_img = Image.open(user_image).resize((300, 300))
    celeb_img = Image.open(celeb_image).resize((300, 300))

    # Create blank canvas
    comparison_img = Image.new('RGB', (650, 400), (20, 20, 20))  # Dark background
    draw = ImageDraw.Draw(comparison_img)

    # Paste images
    comparison_img.paste(user_img, (25, 50))
    comparison_img.paste(celeb_img, (325, 50))

    # Add text
    font = ImageFont.truetype("arial.ttf", 30)
    draw.text((80, 370), "You", font=font, fill=(255, 215, 0))
    draw.text((430, 370), celeb_name, font=font, fill=(255, 215, 0))

    # Custom resemblance text
    resemblance_text = f"You look {resemblance_percentage}% like {celeb_name}"
    draw.text((70, 10), resemblance_text, font=ImageFont.truetype("arial.ttf", 25), fill=(255, 255, 255))

    # Ensure directory exists
    os.makedirs("comparison_results", exist_ok=True)

    # Save Image
    comparison_img_path = os.path.join("comparison_results", "comparison.jpg")
    comparison_img.save(comparison_img_path)
    return comparison_img_path

# Streamlit UI - Bollywood Theme
st.markdown("""
    <style>
        body {background-color: #1a1a1a; color: white; font-family: 'Arial';}
        .stApp {background: linear-gradient(to right, #ff512f, #dd2476);}
        h1 {color: #FFD700; text-align: center; font-size: 40px;}
        .upload-box {border: 2px dashed #FFD700; padding: 20px; text-align: center; border-radius: 15px;}
        .result-name {color: #FFD700; font-size: 28px; font-weight: bold;}
        .image-box {border-radius: 10px; overflow: hidden; box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.2);}
        .center-buttons {display: flex; justify-content: center; gap: 20px; margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎭 Find Your Bollywood Lookalike! 🎬</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload an image and discover which Bollywood star you resemble! 🌟</p>", unsafe_allow_html=True)

# Custom-styled file uploader
uploaded_image = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    st.markdown("<div class='upload-box'>✅ Image uploaded successfully!</div>", unsafe_allow_html=True)

    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        
        # Progress bar for processing
        with st.spinner("🔍 Analyzing your image..."):
            features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        
        if features is None:
            st.stop()
        
        index_pos, resemblance_percentage = recommend(feature_list, features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

        # Generate beautified comparison image
        comparison_image_path = generate_comparison_image(
            os.path.join('uploads', uploaded_image.name), filenames[index_pos], predicted_actor, resemblance_percentage
        )

        # Display Results
        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<h3 style='text-align: center;'>📸 Your Uploaded Image</h3>", unsafe_allow_html=True)
            st.image(display_image, use_container_width=True, caption="Uploaded Image", output_format="PNG")

        with col2:
            st.markdown(f"<h3 style='text-align: center;'>🎬 You Look Like...</h3>", unsafe_allow_html=True)
            st.image(filenames[index_pos], use_container_width=True, caption=predicted_actor, output_format="PNG")
            st.markdown(f"<p class='result-name' style='text-align: center;'>{predicted_actor} ⭐</p>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center; color: #FFD700;'>✨ You look {resemblance_percentage}% like {predicted_actor}</h4>", unsafe_allow_html=True)

        # Download Comparison Image & Share
        st.markdown("---")
        st.markdown("<div class='center-buttons'>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1.5, 1])

        with col1:
            with open(comparison_image_path, "rb") as file:
                st.download_button(label="📥 Download Your Lookalike Image", data=file, file_name="Bollywood_Lookalike.jpg", mime="image/jpeg")

        with col3:
            if st.button("📢 Share Your Lookalike"):
                pyperclip.copy(filenames[index_pos])
                st.success("✅ Image link copied! Paste it to share.")
        
        st.markdown("</div>", unsafe_allow_html=True)
