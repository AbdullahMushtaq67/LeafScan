import os
import json
import datetime
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="LeafScan - Disease Detector & Treatment Lab",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SIDEBAR & THEME SELECTION
# =====================================================
with st.sidebar:
    st.markdown("##  Customization")
    theme_choice = st.selectbox(
        "Select UI Theme",
        ["Crème & Slate (Light)", "Minty Sea Green (Light)", "Ice Blue (Light)", "Dark Slate (Dark)"],
        index=0
    )
    st.divider()

    st.markdown("## About LeafScan")
    st.markdown("""
    LeafScan uses advanced **Deep Learning** (MobileNetV2 CNN) to:
    
    Detect 15 different plant diseases  
    Provide instant treatment recommendations  
    Give confidence scores for predictions  
    Suggest preventive measures  
    
    **Supported Plants:**
    - 🫑 Bell Pepper (2 diseases)
    - Potato (3 diseases)
    - Tomato (10+ diseases)
    """)
    
    st.divider()
    
    st.markdown("##  How It Works")
    st.markdown("""
    1. **Upload** a clear photo of an affected leaf OR **click an example**
    2. **Analyze** with our AI model
    3. **Get insights** on disease type & severity
    4. **Receive treatment** recommendations
    5. **Download PDF Report** or ask the AI Chatbot
    """)
    
    st.divider()
    
    st.markdown("## Disclaimer")
    st.warning("""
    This tool provides **AI-based suggestions only**.  
    For severe plant infections, consult a certified  
    agricultural expert or plant pathologist.
    """)
    
    st.divider()
    st.markdown("<small>Built using TensorFlow & Streamlit</small>", unsafe_allow_html=True)


# =====================================================
# STRICT THEMES, STYLING & ANIMATIONS
# =====================================================
# 1. White/Light themes: ALL text color MUST be solid black (#000000)
# 2. Dark theme: ALL text color MUST be solid white (#FFFFFF)
# 3. NO background is ever pure black (#000000)
if theme_choice == "Crème & Slate (Light)":
    bg_color = "#FAF7F0"
    text_color = "#000000"  # ALL TEXT BLACK
    sidebar_bg = "#F4EFE6"
    card_bg = "#EFEAE0"
    card_border = "#B89B72"
    primary_color = "#B89B72"
    accent_color = "#8D704B"
    border_color = "#DCD5C5"
    input_bg = "#FAF7F0"
elif theme_choice == "Minty Sea Green (Light)":
    bg_color = "#F0FDF4"
    text_color = "#000000"  # ALL TEXT BLACK
    sidebar_bg = "#DCFCE7"
    card_bg = "#D1FAE5"
    card_border = "#10B981"
    primary_color = "#10B981"
    accent_color = "#059669"
    border_color = "#A7F3D0"
    input_bg = "#F0FDF4"
elif theme_choice == "Ice Blue (Light)":
    bg_color = "#F0F9FF"
    text_color = "#000000"  # ALL TEXT BLACK
    sidebar_bg = "#E0F2FE"
    card_bg = "#BAE6FD"
    card_border = "#0284C7"
    primary_color = "#0284C7"
    accent_color = "#0369A1"
    border_color = "#93C5FD"
    input_bg = "#F0F9FF"
else:  # Dark Slate (Dark) - NO PURE BLACK BACKGROUND
    bg_color = "#1E293B"     # Dark Slate Grey (strictly not pure black!)
    text_color = "#FFFFFF"   # ALL TEXT WHITE
    sidebar_bg = "#0F172A"   # Dark Navy Sidebar
    card_bg = "#334155"
    card_border = "#475569"
    primary_color = "#10B981"
    accent_color = "#34D399"
    border_color = "#475569"
    input_bg = "#334155"

# Inject Custom Stylesheet with Animations & STRICT Theme Overrides
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Smooth transition */
    * {{
        transition: background-color 0.4s ease, border-color 0.4s ease;
    }}
    
    /* Global Base */
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
        font-family: 'Poppins', 'Inter', sans-serif !important;
    }}
    
    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        color: {text_color} !important;
    }}
    
    /* STRICT TEXT OVERRIDES: ALL TEXT MUST BE SPECIFIED THEME COLOR */
    h1, h2, h3, h4, h5, h6, p, li, span, label, div, [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] *, .stSelectbox div[data-baseweb="select"] * {{
        color: {text_color} !important;
    }}
    
    .stSelectbox label, .stTextInput label, .stTextArea label, .stChatInput label {{
        color: {text_color} !important;
    }}
    
    .stSelectbox div[data-baseweb="select"] {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        border: 1.5px solid {border_color} !important;
    }}
    
    /* POPOVER / SELECTBOX DROPDOWN STYLING (WHITE BACKGROUND & BLACK TEXT FOREVER) */
    div[data-baseweb="popover"], div[data-baseweb="menu"], [role="option"], [role="listbox"], ul[role="listbox"] {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}
    div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, [role="option"] *, [role="listbox"] * {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}
    [role="option"]:hover, [role="option"][aria-selected="true"] {{
        background-color: #f1f5f9 !important;
        color: #000000 !important;
    }}
    [role="option"]:hover *, [role="option"][aria-selected="true"] * {{
        background-color: #f1f5f9 !important;
        color: #000000 !important;
    }}
    
    [data-testid="stFileUploader"] {{
        background-color: {sidebar_bg} !important;
        border: 1.5px dashed {border_color} !important;
        border-radius: 12px;
        padding: 15px;
    }}
    
    [data-testid="stFileUploader"] * {{
        color: {text_color} !important;
    }}
    
    /* Info cards */
    .info-card {{
        background-color: {card_bg} !important;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid {card_border} !important;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    
    .info-card * {{
        color: {text_color} !important;
    }}
    
    /* Chat Message Boxes styling */
    [data-testid="stChatMessage"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
        padding: 15px !important;
        margin-bottom: 10px !important;
    }}
    [data-testid="stChatMessage"] * {{
        color: {text_color} !important;
    }}
    
    /* Disease warning box */
    .disease-box {{
        background-color: {card_bg} !important;
        padding: 25px;
        border-radius: 12px;
        border: 2px solid {card_border} !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }}
    
    .disease-box * {{
        color: {text_color} !important;
    }}
    
    /* Healthy box */
    .healthy-box {{
        background-color: {card_bg} !important;
        padding: 25px;
        border-radius: 12px;
        border: 2px solid {card_border} !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }}
    
    .healthy-box * {{
        color: {text_color} !important;
    }}
    
    /* Standard Buttons */
    div.stButton button {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1.5px solid {border_color} !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }}
    
    div.stButton button:hover {{
        background-color: {sidebar_bg} !important;
        border-color: {primary_color} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    }}
    
    /* PDF DOWNLOAD BUTTON (WHITE BACKGROUND & BLACK TEXT FOREVER) */
    div.stDownloadButton button, [data-testid="stDownloadButton"] button {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }}
    div.stDownloadButton button:hover, [data-testid="stDownloadButton"] button:hover {{
        background-color: #f1f5f9 !important;
        color: #000000 !important;
        border-color: #000000 !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    }}
    
    /* Logo Banner Styling */
    .logo-banner {{
        background: {sidebar_bg} !important;
        border: 2px solid {border_color} !important;
        border-radius: 20px;
        padding: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }}
    .logo-icon-wrap {{
        background: {card_bg} !important;
        border: 2px solid {border_color} !important;
        border-radius: 50%;
        width: 70px;
        height: 70px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .logo-title-wrap {{
        text-align: left;
    }}
    .logo-title {{
        font-size: 3.2em !important;
        margin: 0 !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 400;
        letter-spacing: -1px;
        line-height: 1;
    }}
    .logo-tagline {{
        font-size: 1.1em !important;
        margin: 8px 0 0 0 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        opacity: 0.95;
    }}
    
    /* Slide-in Animations */
    @keyframes slideInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    .logo-banner, .info-card, .healthy-box, .disease-box, [data-testid="column"] {{
        animation: slideInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}
    
    /* Footer student credit styling */
    .footer-student {{
        text-align: center;
        padding: 20px;
        color: {text_color};
        font-size: 0.85em;
        border-top: 1px solid {border_color};
        margin-top: 50px;
        opacity: 0.8;
    }}
</style>
""", unsafe_allow_html=True)


# =====================================================
# MEDICAL BIOTECH SHIELD LOGO
# =====================================================
st.markdown(f"""
<div class="logo-banner">
    <div class="logo-icon-wrap">
        <svg xmlns="http://www.w3.org/2000/svg" width="45" height="45" viewBox="0 0 24 24" fill="none" stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield"><path d="M20 13c0 5-3.5 7.5-7.66 9.7a1 1 0 0 1-.68 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 .76-.97l8-2a1 1 0 0 1 .48 0l8 2A1 1 0 0 1 20 6v7z"/><path d="M12 17v-8"/><path d="M12 9a4 4 0 0 1 4 4v2c0-2-1.5-3.5-4-3.5"/><path d="M12 11c-2.5 0-4 1.5-4 3.5v-2a4 4 0 0 1 4-4"/></svg>
    </div>
    <div class="logo-title-wrap">
        <h1 class="logo-title">LeafScan <span style="font-weight: 800;">Pathology</span></h1>
        <p class="logo-tagline">AI-Powered Plant Disease Detection & Crop Treatment Guide</p>
    </div>
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
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
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
# EXAMPLE CARD DATA MAPPING
# =====================================================
examples_mapping = {
    "healthy_pepper": {
        "label": "🫑 Pepper Bell (Healthy)",
        "file": "Pepper__bell___healthy1.JPG",
        "category": "Healthy"
    },
    "healthy_potato": {
        "label": "Potato (Healthy)",
        "file": "Potato___healthy1.JPG",
        "category": "Healthy"
    },
    "healthy_tomato": {
        "label": "Tomato (Healthy)",
        "file": "Tomato_healthy1.JPG",
        "category": "Healthy"
    },
    "healthy_potato_alt": {
        "label": "Potato (Healthy Alt)",
        "file": "Potato___healthy2.JPG",
        "category": "Healthy"
    },
    "disease_pepper_spot": {
        "label": "Pepper Bacterial Spot",
        "file": "Pepper__bell___Bacterial_spot1.JPG",
        "category": "Diseased"
    },
    "disease_potato_early": {
        "label": "Potato Early Blight",
        "file": "Potato___Early_blight1.JPG",
        "category": "Diseased"
    },
    "disease_potato_late": {
        "label": "Potato Late Blight",
        "file": "Potato___Late_blight1.JPG",
        "category": "Diseased"
    },
    "disease_tomato_spot": {
        "label": "Tomato Bacterial Spot",
        "file": "Tomato_Bacterial_spot1.JPG",
        "category": "Diseased"
    },
    "disease_tomato_early": {
        "label": "Tomato Early Blight",
        "file": "Tomato_Early_blight1.JPG",
        "category": "Diseased"
    },
    "disease_tomato_late": {
        "label": "Tomato Late Blight",
        "file": "Tomato_Late_blight1.JPG",
        "category": "Diseased"
    }
}


# =====================================================
# PDF REPORT EXPORTER (Requirement 7 Beautified)
# =====================================================
def generate_pdf(prediction, confidence, treatment_info, is_healthy):
    from fpdf import FPDF
    
    # Create PDF object
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Outer frame border layout
    pdf.set_draw_color(16, 185, 129)
    pdf.set_line_width(1.0)
    pdf.rect(5, 5, 200, 287)
    
    # Define Colors
    primary_color = (16, 185, 129)  # Emerald Green
    dark_slate = (15, 23, 42)
    light_slate = (241, 245, 249)
    accent_green = (5, 150, 105)
    text_dark = (30, 41, 59)
    text_muted = (100, 116, 139)
    
    # Header Banner inside border
    pdf.set_fill_color(*primary_color)
    pdf.rect(6, 6, 198, 38, 'F')
    
    # Header Title
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(10, 14)
    pdf.cell(0, 10, "LEAFSCAN PATHOLOGY LAB REPORT", ln=True, align="C")
    
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, "AI-Powered Plant Disease Diagnosis & Prescription", ln=True, align="C")
    pdf.ln(18)
    
    # Reset text color
    pdf.set_text_color(*text_dark)
    
    # Metadata Block
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(12)
    pdf.cell(90, 6, "INSPECTED BY:", ln=False)
    pdf.cell(0, 6, "DATE & TIME OF REPORT:", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*text_muted)
    pdf.set_x(12)
    pdf.cell(90, 6, "Abdullah Mohammad Mushtaq | IQRA University", ln=False)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
    pdf.cell(0, 6, current_time, ln=True)
    pdf.ln(5)
    
    # Divider line
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.4)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(5)
    
    # 1. DIAGNOSIS RESULTS
    pdf.set_text_color(*dark_slate)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_x(12)
    pdf.cell(0, 8, "1. DIAGNOSIS SUMMARY", ln=True)
    pdf.ln(2)
    
    # Extract crop name
    parts = prediction.split("___")
    crop_name = parts[0].replace("_", " ").title() if len(parts) > 1 else prediction.split("_")[0].title()
    disease_display = parts[1].replace("_", " ").title() if len(parts) > 1 else prediction.replace("_", " ").title()
    
    # Background card for diagnosis details
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(12, pdf.get_y(), 186, 26, 'F')
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*text_dark)
    pdf.set_xy(16, pdf.get_y() + 2)
    pdf.cell(45, 7, "Inspected Crop:", ln=False)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, crop_name, ln=True)
    
    pdf.set_x(16)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(45, 7, "AI Diagnosis:", ln=False)
    pdf.set_font("Helvetica", "B", 10)
    if is_healthy:
        pdf.set_text_color(21, 128, 61) # Green
        pdf.cell(0, 7, f"Healthy Plant ({disease_display})", ln=True)
    else:
        pdf.set_text_color(185, 28, 28) # Red
        pdf.cell(0, 7, f"Infected - {disease_display}", ln=True)
        
    pdf.set_text_color(*text_dark)
    pdf.set_x(16)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(45, 7, "Prediction Confidence:", ln=False)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"{confidence:.2f}%", ln=True)
    
    pdf.set_xy(12, pdf.get_y() + 6)
    pdf.ln(3)
    
    # 2. DETAILED PRESCRIPTION
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*dark_slate)
    pdf.cell(0, 8, "2. PATHOLOGY PRESCRIPTION & TREATMENT PLAN", ln=True)
    pdf.ln(2)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(4)
    
    if is_healthy:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*accent_green)
        pdf.set_x(12)
        pdf.cell(0, 7, "PROACTIVE MAINTENANCE INSTRUCTIONS:", ln=True)
        pdf.set_text_color(*text_dark)
        pdf.set_font("Helvetica", "", 10)
        
        tips = [
            "Maintain crop-specific irrigation cycles; avoid waterlogging the roots.",
            "Position the crop to receive the recommended 6-8 hours of sunlight daily.",
            "Verify soil N-P-K nutrient balances and apply slow-release organic fertilizer.",
            "Perform bi-weekly pruning of dry lower leaves to optimize ventilation.",
            f"Pathologist Note: {treatment_info.get('suggestion', 'Plant is in healthy state.')}"
        ]
        for tip in tips:
            pdf.set_x(16)
            pdf.cell(5, 6, "-", ln=False)
            pdf.multi_cell(175, 6, tip)
    else:
        # Box 1: Treatment Method
        pdf.set_fill_color(255, 255, 255)
        pdf.set_draw_color(226, 232, 240)
        pdf.rect(12, pdf.get_y(), 186, 22, 'DF')
        
        pdf.set_xy(15, pdf.get_y() + 1.5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(185, 28, 28)
        pdf.cell(0, 5, "I. RECOMMENDED TREATMENT METHODOLOGY:", ln=True)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*text_dark)
        pdf.set_x(15)
        pdf.multi_cell(180, 5, treatment_info.get('treatment', 'Consult local agricultural division.'))
        
        pdf.set_xy(12, pdf.get_y() + 5)
        
        # Box 2: Recommended Medicine
        pdf.rect(12, pdf.get_y(), 186, 22, 'DF')
        pdf.set_xy(15, pdf.get_y() + 1.5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*accent_green)
        pdf.cell(0, 5, "II. PRESCRIBED MEDICINES / FUNGICIDES:", ln=True)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*text_dark)
        pdf.set_x(15)
        pdf.multi_cell(180, 5, treatment_info.get('medicine', 'N/A'))
        
        pdf.set_xy(12, pdf.get_y() + 5)
        
        # Box 3: Preventative Suggestion
        pdf.rect(12, pdf.get_y(), 186, 22, 'DF')
        pdf.set_xy(15, pdf.get_y() + 1.5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*dark_slate)
        pdf.cell(0, 5, "III. PREVENTATIVE CARE & SUGGESTIONS:", ln=True)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*text_dark)
        pdf.set_x(15)
        pdf.multi_cell(180, 5, treatment_info.get('suggestion', 'N/A'))
        
        pdf.set_xy(12, pdf.get_y() + 6)
        pdf.ln(1)
        
        # Immediate Actions / Biosecurity Warnings (Alert box)
        pdf.set_fill_color(254, 242, 242)
        pdf.set_draw_color(252, 165, 165)
        pdf.rect(12, pdf.get_y(), 186, 32, 'DF')
        
        pdf.set_xy(15, pdf.get_y() + 2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(153, 27, 27)
        pdf.cell(0, 5, "CRITICAL BIO-SECURITY MEASURES & ISOLATION PROTOCOLS", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(127, 29, 29)
        
        measures = [
            "Isolate the infected crop immediately. Spatial proximity will lead to transmission.",
            "Sterilize all cutting/pruning tools with 70% alcohol solution before and after use.",
            "Strictly avoid overhead watering. Leaf wetness accelerates viral and fungal replication."
        ]
        for idx, measure in enumerate(measures):
            pdf.set_x(15)
            pdf.cell(5, 5, f"{idx+1}.", ln=False)
            pdf.cell(0, 5, measure, ln=True)
            
    # Bottom Footer
    pdf.set_y(280)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(203, 213, 225)
    pdf.line(12, 279, 198, 279)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*text_muted)
    pdf.cell(0, 4, "LeafScan AI v2.5 - Agriculture Inspection Tool. Powered by MobileNetV2 Deep Learning.", ln=True, align="C")
    
    # Return PDF as bytes
    pdf_data = pdf.output()
    if isinstance(pdf_data, str):
        return pdf_data.encode('latin1')
    else:
        return bytes(pdf_data)


# =====================================================
# CHATBOT RESPONSE PARSER (Crop + Disease)
# =====================================================
def get_chatbot_response(prompt, treatments):
    prompt_lower = prompt.lower().strip()
    
    # Tokenize and clean prompt
    stopwords = {"how", "do", "i", "treat", "the", "on", "is", "what", "of", "to", "cure", "for", "a", "an", "about", "help", "with", "plant", "leaf", "leaves"}
    words = [w for w in prompt_lower.replace("?", "").replace(".", "").replace(",", "").split() if w not in stopwords]
    
    # Greetings
    if any(greet in prompt_lower for greet in ["hello", "hi", "hey", "hola", "greetings", "pathologist"]):
        return ("Hello! I am the LeafScan Plant Pathologist chatbot. 🍃\n\n"
                "I can help answer questions about plant health, soil conditions, watering practices, "
                "or details of the 15 leaf diseases supported by LeafScan.\n\n"
                "**Try asking something like:**\n"
                "- *'What is early blight?'*\n"
                "- *'Tell me about potato late blight'*\n"
                "- *'How to fertilize tomatoes?'*\n"
                "- *'What plants do you support?'*")

    # Crop/Diseases listing
    if "support" in prompt_lower or "list" in prompt_lower or ("disease" in prompt_lower and len(words) <= 2):
        return ("LeafScan currently supports disease detection and treatment guides for:\n"
                "- **Bell Pepper**: Bacterial Spot, Healthy\n"
                "- **Potato**: Early Blight, Late Blight, Healthy\n"
                "- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, "
                "Spider Mites (Two-spotted), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy\n\n"
                "Ask me about any of these to learn more!")

    # Watering
    if "water" in prompt_lower or "irrigation" in prompt_lower or "watering" in prompt_lower:
        return ("💧 **Watering Best Practices:**\n\n"
                "1. **Water at the Base**: Always water the soil directly, not the leaves. Wet foliage promotes fungal growth like blights and leaf molds.\n"
                "2. **Morning Watering**: Water in the early morning so any moisture on the soil surface can evaporate during the day.\n"
                "3. **Consistency**: Tomatoes and peppers need consistent moisture to prevent blossom end rot and fruit cracking. Soil should feel damp but not waterlogged.\n"
                "4. **Drainage**: Ensure your soil/containers have good drainage to prevent root rot.")
                
    # Soil/pH
    if "soil" in prompt_lower or "ph" in prompt_lower or "acidity" in prompt_lower:
        return ("🌱 **Soil & Nutrition Guide:**\n\n"
                "1. **pH Level**: Tomatoes, peppers, and potatoes prefer slightly acidic to neutral soil (pH 6.0 - 6.8).\n"
                "2. **Nutrients**: Nitrogen (N) promotes leafy growth; Phosphorus (P) helps root development and flowering; Potassium (K) improves overall disease resistance.\n"
                "3. **Organic Matter**: Incorporate compost to improve soil texture, drainage, and microbial activity.\n"
                "4. **Crop Rotation**: Do not plant tomatoes, potatoes, or peppers in the same spot consecutively, as they share similar soil-borne diseases.")

    # Fertilizer
    if "fertiliz" in prompt_lower or "manure" in prompt_lower or "compost" in prompt_lower or "nutrient" in prompt_lower:
        return ("🧪 **Fertilization Advice:**\n\n"
                "1. **Early Stage**: Use balanced N-P-K (e.g., 10-10-10) when planting.\n"
                "2. **Fruit Set**: Once flowers bloom, shift to low-nitrogen, high-phosphorus/potassium formulas (e.g., 5-10-10) to support fruit growth rather than excessive foliage.\n"
                "3. **Calcium Deficiency**: Blossom end rot in tomatoes/peppers is caused by calcium deficiency. Add bone meal or agricultural lime if needed.")

    # Detect crops
    detected_crops = []
    if "pepper" in prompt_lower or "bell" in prompt_lower:
        detected_crops.append("pepper")
    if "potato" in prompt_lower or "potatoes" in prompt_lower:
        detected_crops.append("potato")
    if "tomato" in prompt_lower or "tomatoes" in prompt_lower:
        detected_crops.append("tomato")

    # Detect diseases
    detected_diseases = []
    disease_keywords = {
        "bacterial spot": ["bacterial", "spot", "bacterialspot"],
        "early blight": ["early", "blight", "earlyblight"],
        "late blight": ["late", "blight", "lateblight"],
        "leaf mold": ["mold", "leaf mold", "leaf-mold", "leafmold"],
        "septoria": ["septoria", "septoria leaf spot", "septorialeafspot"],
        "spider mites": ["spider", "mite", "mites", "two-spotted", "twospotted"],
        "target spot": ["target", "target spot", "targetspot"],
        "yellow leaf curl": ["yellow", "curl", "virus", "yellowleaf", "curlvirus"],
        "mosaic virus": ["mosaic", "virus", "mosaicvirus"],
        "healthy": ["healthy", "normal", "fine"]
    }

    for key, kw_list in disease_keywords.items():
        if any(kw in prompt_lower for kw in kw_list):
            detected_diseases.append(key)

    # Search database
    matches = []
    for key in treatments.keys():
        key_lower = key.lower()
        
        # Check crop match
        crop_match = False
        if not detected_crops:
            crop_match = True
        else:
            for crop in detected_crops:
                if crop == "pepper" and "pepper" in key_lower:
                    crop_match = True
                elif crop == "potato" and "potato" in key_lower:
                    crop_match = True
                elif crop == "tomato" and "tomato" in key_lower:
                    crop_match = True

        # Check disease match
        disease_match = False
        if not detected_diseases:
            disease_match = True
        else:
            for disease in detected_diseases:
                norm_key = key_lower.replace("___", " ").replace("_", " ")
                if disease in norm_key:
                    disease_match = True

        if crop_match and disease_match:
            matches.append(key)

    # Output matches
    if matches:
        response = ""
        for match_key in matches[:3]:
            treatment_info = treatments[match_key]
            disease_name = match_key.replace("___", " - ").replace("_", " ").title()
            
            response += f" **Information for {disease_name}:**\n\n"
            if "healthy" in match_key.lower():
                response += f"This is the healthy state for the plant crop. \n\n**Maintenance Tips:** {treatment_info['suggestion']}\n\n"
            else:
                response += (f"**Treatment Method:** {treatment_info['treatment']}\n\n"
                             f"**Recommended Medicine:** {treatment_info['medicine']}\n\n"
                             f"**Additional Suggestions:** {treatment_info['suggestion']}\n\n"
                             "--- \n")
        return response

    return ("I couldn't find a direct match for that query in our plant pathology database. \n\n"
            "Please try specifying the **crop** (Tomato, Potato, or Bell Pepper) and the **disease** (e.g. 'early blight', 'late blight', 'bacterial spot', 'mosaic virus') for a precise diagnosis response.")


# =====================================================
# MAIN APP INTERFACE
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
        <strong> Requirements:</strong>
        • Clear leaf photo<br>
        • Good lighting<br>
        • JPG/PNG format<br>
        • High resolution
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Session State for Clickable Examples
if "selected_example" not in st.session_state:
    st.session_state.selected_example = None

# Reconcile Image Source (Upload vs Clicked Example)
image_to_analyze = None
image_source_name = None

if uploaded_image is not None:
    st.session_state.selected_example = None
    image_to_analyze = Image.open(uploaded_image)
    image_source_name = uploaded_image.name
elif st.session_state.selected_example is not None:
    example_key = st.session_state.selected_example
    if example_key in examples_mapping:
        example_info = examples_mapping[example_key]
        working_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(os.path.dirname(working_dir), "test_images", example_info["file"])
        if os.path.exists(image_path):
            image_to_analyze = Image.open(image_path)
            image_source_name = example_info["label"]

# ANALYSIS AND RESULTS VIEW
if image_to_analyze is not None:
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        st.image(image_to_analyze, use_container_width=True, caption=f"📷 Analyzing: {image_source_name}")
        if st.button(" Clear & Upload Custom Image", use_container_width=True):
            st.session_state.selected_example = None
            st.rerun()
            
    with col2:
        st.subheader(" AI Analysis Results")
        
        with st.spinner(" Analyzing your leaf image..."):
            # Get prediction
            prediction, confidence = predict_image_class(model, image_to_analyze, class_indices)
            
            # Get treatment info
            treatment_info = get_treatment_info(prediction, treatments)
            
            # Check if healthy
            healthy = is_healthy(prediction)
            
            st.markdown("---")
            
            # Display prediction banner
            if healthy:
                st.markdown(f"""
                <div class="healthy-box">
                    <h3> Great News!</h3>
                    <p><strong>Status:</strong> {prediction.replace('_', ' ').replace('healthy', 'Healthy')}</p>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("###  Plant Health Tips")
                st.success(f"""
                **Keep it Healthy!**
                
                 Water regularly based on season  
                 Provide adequate sunlight  
                 Maintain optimal temperature  
                 Use balanced fertilizer monthly  
                 Inspect regularly for early signs  
                
                **Recommendation:** {treatment_info['suggestion']}
                """)
            else:
                st.markdown(f"""
                <div class="disease-box">
                    <h3> Disease Detected</h3>
                    <p><strong>Disease Type:</strong> {prediction.replace('_', ' ').title()}</p>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("###  Treatment Plan")
                
                col_t1, col_t2 = st.columns(2)
                
                with col_t1:
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>Treatment Method:</strong><br>
                        {treatment_info['treatment']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_t2:
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>Recommended Medicine:</strong><br>
                        {treatment_info['medicine']}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("###  Immediate Actions")
                st.error("""
                1. **Isolate** the infected plant immediately
                2. **Remove** all affected leaves/parts
                3. **Disinfect** tools with 70% alcohol
                4. **Avoid** wetting leaves when watering
                5. **Increase** air circulation
                6. **Monitor** nearby plants daily
                """)
                
                st.markdown("###  Additional Suggestion")
                st.info(treatment_info['suggestion'])
            
            # Download PDF Report Section (Beautified Label)
            st.markdown("---")
            st.markdown("###  Export Diagnosis Report")
            try:
                pdf_bytes = generate_pdf(prediction, confidence, treatment_info, healthy)
                st.download_button(
                    label="📥 Download Diagnosis Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"LeafScan_Report_{prediction.replace('___', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as pdf_err:
                st.error(f"Could not generate PDF report: {pdf_err}")

else:
    # LANDING / EXAMPLES VIEW (Requirement 6)
    col_empty1, col_empty2, col_empty3 = st.columns([1, 2, 1])
    
    with col_empty2:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 40px;">
            <h3> Upload a leaf image or click an example below to get started!</h3>
            <p>Our AI will instantly analyze and provide treatment recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # =====================================================
    # NEW FEATURE: INTERACTIVE SYMPTOM DIAGNOSTIC ASSISTANT
    # =====================================================
    with st.expander("🔍 Interactive Symptom Diagnostic Assistant (Optional Pre-Diagnostic)"):
        st.markdown("Check the symptoms you observe on your plant leaf to see the most likely disease matches:")
        
        col_sym1, col_sym2 = st.columns(2)
        with col_sym1:
            sym_yellow = st.checkbox(" Yellow spots or discoloration patches")
            sym_brown = st.checkbox(" Brown or black circular spots")
            sym_wilt = st.checkbox(" Leaf wilting or drooping stems")
        with col_sym2:
            sym_webs = st.checkbox(" Fine webbing on leaf undersides")
            sym_curl = st.checkbox(" Curled, crinkled, or deformed leaf shape")
            sym_mold = st.checkbox(" White, grey, or velvety powdery coating")
            
        selected_symptoms = []
        if sym_yellow: selected_symptoms.append("yellow")
        if sym_brown: selected_symptoms.append("brown")
        if sym_wilt: selected_symptoms.append("wilt")
        if sym_webs: selected_symptoms.append("webs")
        if sym_curl: selected_symptoms.append("curl")
        if sym_mold: selected_symptoms.append("mold")
        
        if selected_symptoms:
            st.markdown("##### 🔬 Potential Disease Matches:")
            # Define symptom mapping for each disease class
            disease_symptoms = {
                "Pepper__bell___Bacterial_spot": ["yellow", "brown"],
                "Potato___Early_blight": ["brown"],
                "Potato___Late_blight": ["brown", "wilt"],
                "Tomato_Bacterial_spot": ["yellow", "brown"],
                "Tomato_Early_blight": ["brown", "yellow"],
                "Tomato_Late_blight": ["brown", "wilt"],
                "Tomato_Leaf_Mold": ["yellow", "mold"],
                "Tomato_Septoria_leaf_spot": ["brown"],
                "Tomato_Spider_mites_Two_spotted_spider_mite": ["yellow", "webs"],
                "Tomato__Target_Spot": ["brown"],
                "Tomato__Tomato_YellowLeaf__Curl_Virus": ["yellow", "curl"],
                "Tomato__Tomato_mosaic_virus": ["yellow", "curl"]
            }
            
            matches = []
            for dis_class, syms in disease_symptoms.items():
                intersection = set(selected_symptoms).intersection(set(syms))
                score = len(intersection) / len(syms) if syms else 0
                if score > 0:
                    matches.append((dis_class, score))
                    
            if matches:
                matches.sort(key=lambda x: x[1], reverse=True)
                for item in matches[:3]:
                    dis_name = item[0].replace("___", " - ").replace("_", " ").title()
                    match_percentage = item[1] * 100
                    st.write(f"**{dis_name}** ({match_percentage:.0f}% match)")
                    st.progress(item[1])
            else:
                st.info("No matching diseases found for the selected symptoms. Try uploading a photo for AI analysis!")
        else:
            st.info("Select one or more symptoms above to see live diagnostic estimations.")

    st.markdown("---")
    st.subheader("💡 Quick Test: Click an Example Leaf Image")
    st.markdown("Click one of the 4 healthy or 6 diseased leaves below to run disease detection instantly:")

    st.markdown("####  Correct (Healthy) Leaves (4 Examples)")
    cols_healthy = st.columns(4)
    healthy_examples = [item for item in examples_mapping.items() if item[1]["category"] == "Healthy"]
    for idx, (key, info) in enumerate(healthy_examples):
        with cols_healthy[idx]:
            working_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(os.path.dirname(working_dir), "test_images", info["file"])
            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
                if st.button(info["label"], key=key, use_container_width=True):
                    st.session_state.selected_example = key
                    st.rerun()
            else:
                st.warning(f"{info['file']} not found")

    st.markdown("####  Diseased Leaves (6 Examples)")
    cols_diseased = st.columns(3)
    diseased_examples = [item for item in examples_mapping.items() if item[1]["category"] == "Diseased"]
    for idx, (key, info) in enumerate(diseased_examples):
        col_idx = idx % 3
        with cols_diseased[col_idx]:
            working_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(os.path.dirname(working_dir), "test_images", info["file"])
            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
                if st.button(info["label"], key=key, use_container_width=True):
                    st.session_state.selected_example = key
                    st.rerun()
            else:
                st.warning(f"{info['file']} not found")


# =====================================================
# AI CHATBOT SECTION (Requirement 3)
# =====================================================
st.markdown("---")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "👋 Hello! I'm your LeafScan AI assistant. Ask me anything about plant diseases, treatments, or plant care."}
    ]

col_chat_title, col_chat_clear = st.columns([4, 1])
with col_chat_title:
    st.markdown("###  LeafScan Plant Pathology Chatbot")
with col_chat_clear:
    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "👋 Hello! I'm your LeafScan AI assistant. Ask me anything about plant diseases, treatments, or plant care."}
        ]
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question (e.g., 'What is Early Blight?' or 'How do I care for tomatoes?')"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Thinking..."):
            response = get_chatbot_response(prompt, treatments)
        response_placeholder.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})


# =====================================================
# FOOTER (Requirement 5)
# =====================================================
st.markdown(f"""
<div class="footer-student">
    <p>🍃 <b>LeafScan v2.5</b> | AI-Powered Plant Pathology Lab</p>
    <p>Built with TensorFlow, Keras & Streamlit | Powered by MobileNetV2 CNN</p>
    <p><b>Abdullah Mohammad Mushtaq | IQRA University</b></p>
</div>
""", unsafe_allow_html=True)