import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="LeafScan - Disease Detector",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(46, 204, 113, 0.3);
    }
    
    .main-title h1 {
        font-size: 3em;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-title p {
        font-size: 1.1em;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2ecc71;
        margin: 15px 0;
        color: #1b5e20;
        font-weight: 500;
    }
    
    /* Disease warning box */
    .disease-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        border: 2px solid #c92a2a;
    }
    
    /* Healthy box */
    .healthy-box {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        border: 2px solid #2f9e44;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: #7f8c8d;
        font-size: 0.9em;
        border-top: 1px solid #ecf0f1;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-title">
    <h1>🍃 LeafScan</h1>
    <p>AI-Powered Plant Disease Detection & Treatment Guide</p>
</div>
""", unsafe_allow_html=True)

st.markdown("Instantly identify plant diseases from leaf images and get personalized treatment recommendations.", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL AND DATA
# =====================================================
@st.cache_resource
def load_model_and_data():
    """Load model, class indices, and treatments (cached for performance)"""
    working_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Rebuild model architecture (same as training notebook)
    from tensorflow.keras.applications import MobileNetV2
    
    img_size = 224
    base = MobileNetV2(input_shape=(img_size, img_size, 3),
                       include_top=False,
                       weights='imagenet')
    base.trainable = False
    
    model = tf.keras.models.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(15, activation='softmax')  # 15 classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Load weights from .h5 file
    model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
    try:
        model.load_weights(model_path)
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        raise
    
    # Load class indices
    class_indices_path = os.path.join(working_dir, "class_indices.json")
    class_indices = json.load(open(class_indices_path))
    
    # Load treatments
    treatments_path = os.path.join(working_dir, "treatments.json")
    treatments = json.load(open(treatments_path))
    
    return model, class_indices, treatments


model, class_indices, treatments = load_model_and_data()


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def load_and_preprocess_image(image, target_size=(224, 224)):
    """Load and preprocess image for prediction"""
    # Convert to RGB (removes alpha/transparency channel from PNG)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image
    img = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Scale to [0, 1]
    img_array = img_array.astype('float32') / 255.
    
    return img_array


def predict_image_class(model, image, class_indices):
    """Predict disease class and return prediction with confidence"""
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence


def get_treatment_info(disease_name, treatments):
    """Get treatment information for a disease"""
    if disease_name in treatments:
        return treatments[disease_name]
    return {
        "treatment": "Consult agricultural expert",
        "medicine": "N/A",
        "suggestion": "No data available for this disease"
    }


def is_healthy(disease_name):
    """Check if the plant is healthy"""
    return "healthy" in disease_name.lower()


# =====================================================
# SIDEBAR - INFORMATION
# =====================================================
with st.sidebar:
    st.markdown("## ℹ️ About LeafScan")
    st.markdown("""
    LeafScan uses advanced **Deep Learning** (MobileNetV2 CNN) to:
    
    ✅ Detect 15 different plant diseases  
    ✅ Provide instant treatment recommendations  
    ✅ Give confidence scores for predictions  
    ✅ Suggest preventive measures  
    
    **Supported Plants:**
    - 🫑 Bell Pepper (2 diseases)
    - 🥔 Potato (3 diseases)
    - 🍅 Tomato (10+ diseases)
    """)
    
    st.divider()
    
    st.markdown("## 🔧 How It Works")
    st.markdown("""
    1. **Upload** a clear photo of an affected leaf
    2. **Analyze** with our AI model
    3. **Get insights** on disease type & severity
    4. **Receive treatment** recommendations
    5. **Follow-up** monitoring tips
    """)
    
    st.divider()
    
    st.markdown("## ⚠️ Disclaimer")
    st.warning("""
    This tool provides **AI-based suggestions only**.  
    For severe plant infections, consult a certified  
    agricultural expert or plant pathologist.
    """)
    
    st.divider()
    
    st.markdown("<small>Built with 🤍 using TensorFlow & Streamlit</small>", unsafe_allow_html=True)


# =====================================================
# MAIN APP
# =====================================================
st.markdown("---")

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("📸 Upload Your Leaf Image")
    st.markdown("*For best results: clear photo, good lighting, focused on affected areas*")
    uploaded_image = st.file_uploader(
        "Choose a JPG or PNG image", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("""
    <div class="info-card">
        <strong>📋 Requirements:</strong>
        • Clear leaf photo
        • Good lighting
        • JPG/PNG format
        • High resolution
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# If image is uploaded
if uploaded_image is not None:
    # Display image and results side by side
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        image = Image.open(uploaded_image)
        st.image(image, use_container_width=True, caption="📷 Your Leaf Image")
    
    with col2:
        st.subheader("🔬 AI Analysis Results")
        
        if st.button("🚀 Analyze with LeafScan", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing your leaf image..."):
                # Get prediction
                prediction, confidence = predict_image_class(model, image, class_indices)
                
                # Get treatment info
                treatment_info = get_treatment_info(prediction, treatments)
                
                # Check if healthy
                healthy = is_healthy(prediction)
                
                st.markdown("---")
                
                # Display prediction
                if healthy:
                    st.markdown(f"""
                    <div class="healthy-box">
                        <h3>✅ Great News!</h3>
                        <p><strong>Status:</strong> {prediction.replace('_', ' ')}</p>
                        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 🌱 Plant Health Tips")
                    st.success(f"""
                    **Keep it Healthy!**
                    
                    💧 Water regularly based on season  
                    ☀️ Provide adequate sunlight  
                    🌡️ Maintain optimal temperature  
                    🧪 Use balanced fertilizer monthly  
                    🔍 Inspect regularly for early signs  
                    
                    **Recommendation:** {treatment_info['suggestion']}
                    """)
                else:
                    st.markdown(f"""
                    <div class="disease-box">
                        <h3>⚠️ Disease Detected</h3>
                        <p><strong>Disease Type:</strong> {prediction.replace('_', ' ').title()}</p>
                        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 💊 Treatment Plan")
                    
                    col_t1, col_t2 = st.columns(2)
                    
                    with col_t1:
                        st.markdown("""
                        <div class="info-card">
                            <strong>Treatment Method:</strong><br>
                        """, unsafe_allow_html=True)
                        st.write(treatment_info['treatment'])
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col_t2:
                        st.markdown("""
                        <div class="info-card">
                            <strong>Recommended Medicine:</strong><br>
                        """, unsafe_allow_html=True)
                        st.write(treatment_info['medicine'])
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("### 🚨 Immediate Actions")
                    st.error("""
                    1. **Isolate** the infected plant immediately
                    2. **Remove** all affected leaves/parts
                    3. **Disinfect** tools with 70% alcohol
                    4. **Avoid** wetting leaves when watering
                    5. **Increase** air circulation
                    6. **Monitor** nearby plants daily
                    """)
                    
                    st.markdown("### 💡 Additional Suggestion")
                    st.info(treatment_info['suggestion'])

else:
    # Landing message
    col_empty1, col_empty2, col_empty3 = st.columns([1, 2, 1])
    
    with col_empty2:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 40px;">
            <h3>👆 Upload a leaf image to get started!</h3>
            <p>Our AI will instantly analyze and provide treatment recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Example images section
    st.markdown("### 📸 Example Images")
    st.markdown("""
    For best results, ensure your image:
    - ✅ Shows a clear view of the leaf
    - ✅ Has good lighting
    - ✅ Focuses on affected areas
    - ✅ Is in JPG, JPEG, or PNG format
    """)


# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    <p><strong>🍃 LeafScan v2.0</strong> | AI-Powered Plant Disease Detection</p>
    <p>Built with TensorFlow, Keras & Streamlit | Powered by MobileNetV2 CNN</p>
    <p style="font-size: 0.85em; color: #95a5a6;">Enter the future of plant care 🚀</p>
</div>
""", unsafe_allow_html=True)