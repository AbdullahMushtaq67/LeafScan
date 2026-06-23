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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# SIDEBAR & THEME SELECTION
# =====================================================
with st.sidebar:
    st.markdown("## Interface Settings")
    theme_choice = st.selectbox(
        "Select UI Theme",
        ["Creme & Slate (Light)", "Minty Sea Green (Light)", "Ice Blue (Light)", "Dark Slate (Dark)"],
        index=0
    )
    st.divider()

    st.markdown("## About LeafScan")
    st.markdown("""
    LeafScan utilizes advanced **Deep Learning** (MobileNetV2 CNN) to:

    - Detect 15 distinct plant diseases
    - Provide instant treatment recommendations
    - Generate confidence scores for each prediction
    - Suggest preventive and maintenance measures

    **Supported Plant Species:**
    - Bell Pepper (2 disease classes)
    - Potato (3 disease classes)
    - Tomato (10+ disease classes)
    """)

    st.divider()

    st.markdown("## How It Works")
    st.markdown("""
    1. **Upload** a clear photograph of an affected leaf, or select a provided example
    2. **Submit** the image for AI-based analysis
    3. **Review** the diagnosis, disease type, and confidence score
    4. **Follow** the prescribed treatment recommendations
    5. **Download** a formal PDF report for your records
    """)

    st.divider()

    st.markdown("## Disclaimer")
    st.warning("""
    This tool provides **AI-based suggestions only**.
    For severe or persistent plant infections, consult a certified
    agricultural expert or qualified plant pathologist.
    """)

    st.divider()
    st.markdown("<small>Powered by TensorFlow and Streamlit</small>", unsafe_allow_html=True)


# =====================================================
# THEME CONFIGURATION
# Light themes: text color is solid black (#000000)
# Dark theme:  text color is solid white (#FFFFFF)
# No background uses pure black (#000000)
# =====================================================
if theme_choice == "Creme & Slate (Light)":
    bg_color = "#FAF7F0"
    text_color = "#000000"
    sidebar_bg = "#F4EFE6"
    card_bg = "#EFEAE0"
    card_border = "#B89B72"
    primary_color = "#B89B72"
    accent_color = "#8D704B"
    border_color = "#DCD5C5"
    input_bg = "#FAF7F0"
    is_dark = False
elif theme_choice == "Minty Sea Green (Light)":
    bg_color = "#F0FDF4"
    text_color = "#000000"
    sidebar_bg = "#DCFCE7"
    card_bg = "#D1FAE5"
    card_border = "#10B981"
    primary_color = "#10B981"
    accent_color = "#059669"
    border_color = "#A7F3D0"
    input_bg = "#F0FDF4"
    is_dark = False
elif theme_choice == "Ice Blue (Light)":
    bg_color = "#F0F9FF"
    text_color = "#000000"
    sidebar_bg = "#E0F2FE"
    card_bg = "#BAE6FD"
    card_border = "#0284C7"
    primary_color = "#0284C7"
    accent_color = "#0369A1"
    border_color = "#93C5FD"
    input_bg = "#F0F9FF"
    is_dark = False
else:  # Dark Slate
    bg_color = "#1E293B"
    text_color = "#FFFFFF"
    sidebar_bg = "#0F172A"
    card_bg = "#334155"
    card_border = "#475569"
    primary_color = "#10B981"
    accent_color = "#34D399"
    border_color = "#475569"
    input_bg = "#334155"
    is_dark = True

# Button text is always the theme text color (white on dark, black on light)
btn_text_color = text_color

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    * {{
        transition: background-color 0.4s ease, border-color 0.4s ease;
    }}

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

    h1, h2, h3, h4, h5, h6, p, li, span, label, div,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] *,
    .stSelectbox div[data-baseweb="select"] * {{
        color: {text_color} !important;
    }}

    .stSelectbox label, .stTextInput label, .stTextArea label, .stChatInput label {{
        color: {text_color} !important;
    }}

    /* Selectbox selected value — must follow theme text color */
    .stSelectbox div[data-baseweb="select"] {{
        background-color: {input_bg} !important;
        border: 1.5px solid {border_color} !important;
    }}
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div {{
        color: {text_color} !important;
        background-color: {input_bg} !important;
    }}
    /* Dropdown arrow icon */
    .stSelectbox svg {{
        fill: {text_color} !important;
    }}

    /* Dropdown popover always uses white background with black text for legibility */
    div[data-baseweb="popover"], div[data-baseweb="menu"],
    [role="option"], [role="listbox"], ul[role="listbox"] {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}
    div[data-baseweb="popover"] *, div[data-baseweb="menu"] *,
    [role="option"] *, [role="listbox"] * {{
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

    /* Standard buttons — theme-aware text color */
    div.stButton button {{
        background-color: {card_bg} !important;
        color: {btn_text_color} !important;
        border: 1.5px solid {border_color} !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }}
    div.stButton button *,
    div.stButton button p,
    div.stButton button span,
    div.stButton button div,
    div.stButton > button > div > p {{
        color: {btn_text_color} !important;
        background-color: transparent !important;
    }}
    div.stButton button:hover {{
        background-color: {sidebar_bg} !important;
        border-color: {primary_color} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    }}

    /* Download button — theme-aware text color */
    div.stDownloadButton button,
    [data-testid="stDownloadButton"] button {{
        background-color: {card_bg} !important;
        color: {btn_text_color} !important;
        border: 2px solid {border_color} !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }}
    div.stDownloadButton button *,
    div.stDownloadButton button p,
    div.stDownloadButton button span,
    div.stDownloadButton button div,
    [data-testid="stDownloadButton"] button *,
    [data-testid="stDownloadButton"] button p,
    [data-testid="stDownloadButton"] button span,
    [data-testid="stDownloadButton"] button div {{
        color: {btn_text_color} !important;
        background-color: transparent !important;
    }}
    div.stDownloadButton button:hover,
    [data-testid="stDownloadButton"] button:hover {{
        background-color: {sidebar_bg} !important;
        border-color: {primary_color} !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    }}

    /* File uploader Browse button */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {card_bg} !important;
        color: {btn_text_color} !important;
        border: 1.5px solid {border_color} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    [data-testid="stFileUploader"] button *,
    [data-testid="stFileUploader"] button p,
    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzone"] button span {{
        color: {btn_text_color} !important;
        background-color: transparent !important;
    }}

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

    @keyframes slideInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .logo-banner, .info-card, .healthy-box, .disease-box, [data-testid="column"] {{
        animation: slideInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}

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
# LOGO BANNER
# =====================================================
st.markdown(f"""
<div class="logo-banner">
    <div class="logo-icon-wrap">
        <svg xmlns="http://www.w3.org/2000/svg" width="45" height="45" viewBox="0 0 24 24" fill="none"
             stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20 13c0 5-3.5 7.5-7.66 9.7a1 1 0 0 1-.68 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 .76-.97l8-2a1 1 0 0 1 .48 0l8 2A1 1 0 0 1 20 6v7z"/>
            <path d="M12 17v-8"/>
            <path d="M12 9a4 4 0 0 1 4 4v2c0-2-1.5-3.5-4-3.5"/>
            <path d="M12 11c-2.5 0-4 1.5-4 3.5v-2a4 4 0 0 1 4-4"/>
        </svg>
    </div>
    <div class="logo-title-wrap">
        <h1 class="logo-title">LeafScan <span style="font-weight: 800;">Pathology</span></h1>
        <p class="logo-tagline">AI-Powered Plant Disease Detection &amp; Crop Treatment Guide</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Upload a leaf photograph to receive an instant AI-based disease diagnosis and personalized treatment recommendations.",
    unsafe_allow_html=True
)


# =====================================================
# LOAD MODEL AND DATA
# =====================================================
@st.cache_resource
def load_model_and_data():
    """Load model, class indices, and treatments (cached for performance)."""
    working_dir = os.path.dirname(os.path.abspath(__file__))

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
        tf.keras.layers.Dense(15, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
    try:
        model.load_weights(model_path)
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        raise

    class_indices_path = os.path.join(working_dir, "class_indices.json")
    class_indices = json.load(open(class_indices_path))

    treatments_path = os.path.join(working_dir, "treatments.json")
    treatments = json.load(open(treatments_path))

    return model, class_indices, treatments


model, class_indices, treatments = load_model_and_data()


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def load_and_preprocess_image(image, target_size=(224, 224)):
    """Convert, resize, and normalize an image for model inference."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


def predict_image_class(model, image, class_indices):
    """Run inference and return the predicted class name and confidence score."""
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence


def get_treatment_info(disease_name, treatments):
    """Retrieve treatment information for a given disease name."""
    if disease_name in treatments:
        return treatments[disease_name]
    return {
        "treatment": "Consult a qualified agricultural expert.",
        "medicine": "N/A",
        "suggestion": "No treatment data is available for this classification."
    }


def is_healthy(disease_name):
    """Return True if the classification indicates a healthy plant."""
    return "healthy" in disease_name.lower()


# =====================================================
# EXAMPLE IMAGES MAPPING
# =====================================================
examples_mapping = {
    "healthy_pepper": {
        "label": "Bell Pepper — Healthy",
        "file": "Pepper__bell___healthy1.JPG",
        "category": "Healthy"
    },
    "healthy_potato": {
        "label": "Potato — Healthy",
        "file": "Potato___healthy1.JPG",
        "category": "Healthy"
    },
    "healthy_tomato": {
        "label": "Tomato — Healthy",
        "file": "Tomato_healthy1.JPG",
        "category": "Healthy"
    },
    "healthy_potato_alt": {
        "label": "Potato — Healthy (Alt.)",
        "file": "Potato___healthy2.JPG",
        "category": "Healthy"
    },
    "disease_pepper_spot": {
        "label": "Bell Pepper — Bacterial Spot",
        "file": "Pepper__bell___Bacterial_spot1.JPG",
        "category": "Diseased"
    },
    "disease_potato_early": {
        "label": "Potato — Early Blight",
        "file": "Potato___Early_blight1.JPG",
        "category": "Diseased"
    },
    "disease_potato_late": {
        "label": "Potato — Late Blight",
        "file": "Potato___Late_blight1.JPG",
        "category": "Diseased"
    },
    "disease_tomato_spot": {
        "label": "Tomato — Bacterial Spot",
        "file": "Tomato_Bacterial_spot1.JPG",
        "category": "Diseased"
    },
    "disease_tomato_early": {
        "label": "Tomato — Early Blight",
        "file": "Tomato_Early_blight1.JPG",
        "category": "Diseased"
    },
    "disease_tomato_late": {
        "label": "Tomato — Late Blight",
        "file": "Tomato_Late_blight1.JPG",
        "category": "Diseased"
    }
}


# =====================================================
# PDF REPORT GENERATOR
# =====================================================
def generate_pdf(prediction, confidence, treatment_info, is_healthy):
    from fpdf import FPDF

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_draw_color(16, 185, 129)
    pdf.set_line_width(1.0)
    pdf.rect(5, 5, 200, 287)

    primary_color = (16, 185, 129)
    dark_slate = (15, 23, 42)
    accent_green = (5, 150, 105)
    text_dark = (30, 41, 59)
    text_muted = (100, 116, 139)

    pdf.set_fill_color(*primary_color)
    pdf.rect(6, 6, 198, 38, 'F')

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(10, 14)
    pdf.cell(0, 10, "LEAFSCAN PATHOLOGY LAB REPORT", ln=True, align="C")

    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, "AI-Powered Plant Disease Diagnosis & Prescription", ln=True, align="C")
    pdf.ln(18)

    pdf.set_text_color(*text_dark)

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

    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.4)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(5)

    pdf.set_text_color(*dark_slate)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_x(12)
    pdf.cell(0, 8, "1. DIAGNOSIS SUMMARY", ln=True)
    pdf.ln(2)

    parts = prediction.split("___")
    crop_name = parts[0].replace("_", " ").title() if len(parts) > 1 else prediction.split("_")[0].title()
    disease_display = parts[1].replace("_", " ").title() if len(parts) > 1 else prediction.replace("_", " ").title()

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
    if is_healthy:
        pdf.set_text_color(21, 128, 61)
        pdf.cell(0, 7, f"Healthy Plant ({disease_display})", ln=True)
    else:
        pdf.set_text_color(185, 28, 28)
        pdf.cell(0, 7, f"Infected - {disease_display}", ln=True)

    pdf.set_text_color(*text_dark)
    pdf.set_x(16)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(45, 7, "Prediction Confidence:", ln=False)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"{confidence:.2f}%", ln=True)

    pdf.set_xy(12, pdf.get_y() + 6)
    pdf.ln(3)

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
            f"Pathologist Note: {treatment_info.get('suggestion', 'Plant is in a healthy state.')}"
        ]
        for tip in tips:
            pdf.set_x(16)
            pdf.cell(5, 6, "-", ln=False)
            pdf.multi_cell(175, 6, tip)
    else:
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
            "Sterilize all cutting and pruning tools with 70% alcohol solution before and after use.",
            "Strictly avoid overhead watering. Leaf wetness accelerates viral and fungal replication."
        ]
        for idx, measure in enumerate(measures):
            pdf.set_x(15)
            pdf.cell(5, 5, f"{idx + 1}.", ln=False)
            pdf.cell(0, 5, measure, ln=True)

    pdf.set_y(280)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(203, 213, 225)
    pdf.line(12, 279, 198, 279)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*text_muted)
    pdf.cell(0, 4, "LeafScan AI v2.5 - Agriculture Inspection Tool. Powered by MobileNetV2 Deep Learning.", ln=True, align="C")

    pdf_data = pdf.output()
    if isinstance(pdf_data, str):
        return pdf_data.encode('latin1')
    else:
        return bytes(pdf_data)


# =====================================================
# CHATBOT RESPONSE PARSER
# =====================================================
def get_chatbot_response(prompt, treatments):
    prompt_lower = prompt.lower().strip()

    stopwords = {
        "how", "do", "i", "treat", "the", "on", "is", "what", "of", "to",
        "cure", "for", "a", "an", "about", "help", "with", "plant", "leaf", "leaves"
    }
    words = [
        w for w in prompt_lower.replace("?", "").replace(".", "").replace(",", "").split()
        if w not in stopwords
    ]

    if any(greet in prompt_lower for greet in ["hello", "hi", "hey", "hola", "greetings", "pathologist"]):
        return (
            "Welcome to the LeafScan Plant Pathology Assistant.\n\n"
            "I am able to answer questions regarding plant health, soil conditions, irrigation practices, "
            "and the 15 leaf disease classes supported by LeafScan.\n\n"
            "**Example queries:**\n"
            "- *What is Early Blight?*\n"
            "- *Describe Potato Late Blight.*\n"
            "- *How should tomatoes be fertilized?*\n"
            "- *Which plant species are supported?*"
        )

    if "support" in prompt_lower or "list" in prompt_lower or ("disease" in prompt_lower and len(words) <= 2):
        return (
            "LeafScan currently supports disease detection and treatment guidance for the following:\n\n"
            "- **Bell Pepper**: Bacterial Spot, Healthy\n"
            "- **Potato**: Early Blight, Late Blight, Healthy\n"
            "- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, "
            "Two-Spotted Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy\n\n"
            "Please specify a disease name for detailed treatment information."
        )

    if "water" in prompt_lower or "irrigation" in prompt_lower or "watering" in prompt_lower:
        return (
            "**Irrigation Best Practices:**\n\n"
            "1. **Base Watering:** Apply water directly to the soil, not the foliage. Wet leaves promote fungal diseases such as blight and leaf mold.\n"
            "2. **Morning Application:** Water in the early morning to allow surface moisture to evaporate during daylight hours.\n"
            "3. **Consistent Moisture:** Tomatoes and peppers require consistent soil moisture to prevent blossom-end rot and fruit cracking. The soil should remain damp but well-drained.\n"
            "4. **Adequate Drainage:** Ensure containers and beds have sufficient drainage to prevent root rot."
        )

    if "soil" in prompt_lower or "ph" in prompt_lower or "acidity" in prompt_lower:
        return (
            "**Soil and Nutrition Guidelines:**\n\n"
            "1. **pH Level:** Tomatoes, peppers, and potatoes prefer slightly acidic to neutral soil (pH 6.0 - 6.8).\n"
            "2. **Macronutrients:** Nitrogen (N) supports foliar growth; Phosphorus (P) aids root development and flowering; Potassium (K) improves overall disease resistance.\n"
            "3. **Organic Matter:** Incorporate compost to improve soil texture, drainage, and microbial activity.\n"
            "4. **Crop Rotation:** Avoid planting tomatoes, potatoes, or peppers in the same location consecutively, as these share common soil-borne diseases."
        )

    if "fertiliz" in prompt_lower or "manure" in prompt_lower or "compost" in prompt_lower or "nutrient" in prompt_lower:
        return (
            "**Fertilization Guidance:**\n\n"
            "1. **Establishment Stage:** Apply a balanced N-P-K formula (e.g., 10-10-10) at the time of planting.\n"
            "2. **Fruit Set Stage:** After flowering begins, transition to a low-nitrogen, high-phosphorus/potassium formula (e.g., 5-10-10) to support fruit development over vegetative growth.\n"
            "3. **Calcium Deficiency:** Blossom-end rot in tomatoes and peppers is associated with calcium deficiency. Apply bone meal or agricultural lime as needed."
        )

    detected_crops = []
    if "pepper" in prompt_lower or "bell" in prompt_lower:
        detected_crops.append("pepper")
    if "potato" in prompt_lower or "potatoes" in prompt_lower:
        detected_crops.append("potato")
    if "tomato" in prompt_lower or "tomatoes" in prompt_lower:
        detected_crops.append("tomato")

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

    matches = []
    for key in treatments.keys():
        key_lower = key.lower()

        crop_match = not detected_crops
        if not crop_match:
            for crop in detected_crops:
                if crop == "pepper" and "pepper" in key_lower:
                    crop_match = True
                elif crop == "potato" and "potato" in key_lower:
                    crop_match = True
                elif crop == "tomato" and "tomato" in key_lower:
                    crop_match = True

        disease_match = not detected_diseases
        if not disease_match:
            for disease in detected_diseases:
                norm_key = key_lower.replace("___", " ").replace("_", " ")
                if disease in norm_key:
                    disease_match = True

        if crop_match and disease_match:
            matches.append(key)

    if matches:
        response = ""
        for match_key in matches[:3]:
            treatment_info = treatments[match_key]
            disease_name = match_key.replace("___", " - ").replace("_", " ").title()

            response += f"**Pathology Information: {disease_name}**\n\n"
            if "healthy" in match_key.lower():
                response += (
                    f"This classification represents a healthy plant.\n\n"
                    f"**Maintenance Recommendation:** {treatment_info['suggestion']}\n\n"
                )
            else:
                response += (
                    f"**Treatment Method:** {treatment_info['treatment']}\n\n"
                    f"**Recommended Medicine:** {treatment_info['medicine']}\n\n"
                    f"**Additional Notes:** {treatment_info['suggestion']}\n\n"
                    "---\n"
                )
        return response

    return (
        "No direct match was found in the plant pathology database for that query.\n\n"
        "Please specify the **crop** (Tomato, Potato, or Bell Pepper) and the **disease** "
        "(e.g., Early Blight, Late Blight, Bacterial Spot, Mosaic Virus) for a precise response."
    )


# =====================================================
# MAIN APP INTERFACE
# =====================================================
st.markdown("---")

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("Upload Leaf Image")
    st.markdown("*For optimal results: submit a clear, well-lit photograph focused on the affected leaf area.*")
    uploaded_image = st.file_uploader(
        "Select a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("""
    <div class="info-card">
        <strong>Image Requirements:</strong><br>
        - Clear, focused leaf photograph<br>
        - Adequate lighting<br>
        - JPG or PNG format<br>
        - High resolution preferred
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if "selected_example" not in st.session_state:
    st.session_state.selected_example = None

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


# =====================================================
# ANALYSIS AND RESULTS VIEW
# =====================================================
if image_to_analyze is not None:
    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        st.image(image_to_analyze, use_container_width=True, caption=f"Submitted Image: {image_source_name}")
        if st.button("Clear Image and Upload New", use_container_width=True):
            st.session_state.selected_example = None
            st.rerun()

    with col2:
        st.subheader("AI Analysis Results")

        with st.spinner("Analyzing submitted image. Please wait..."):
            prediction, confidence = predict_image_class(model, image_to_analyze, class_indices)
            treatment_info = get_treatment_info(prediction, treatments)
            healthy = is_healthy(prediction)

            st.markdown("---")

            if healthy:
                st.markdown(f"""
                <div class="healthy-box">
                    <h3>Diagnosis: Healthy</h3>
                    <p><strong>Classification:</strong> {prediction.replace('_', ' ').replace('healthy', 'Healthy')}</p>
                    <p><strong>Confidence Score:</strong> {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Maintenance Recommendations")
                st.success(f"""
                **The plant appears healthy. Continue the following practices:**

                - Water regularly according to seasonal requirements
                - Ensure adequate daily sunlight exposure
                - Maintain optimal ambient temperature
                - Apply a balanced fertilizer on a monthly schedule
                - Conduct routine inspections for early signs of disease

                **Pathologist Note:** {treatment_info['suggestion']}
                """)
            else:
                st.markdown(f"""
                <div class="disease-box">
                    <h3>Disease Detected</h3>
                    <p><strong>Classification:</strong> {prediction.replace('_', ' ').title()}</p>
                    <p><strong>Confidence Score:</strong> {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Prescribed Treatment Plan")

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

                st.markdown("### Immediate Actions Required")
                st.error("""
                1. **Isolate** the infected plant from surrounding crops immediately
                2. **Remove** all visibly infected leaves and plant material
                3. **Disinfect** all cutting and pruning tools with 70% alcohol solution
                4. **Avoid** overhead watering; apply water at the base only
                5. **Improve** air circulation around the affected plant
                6. **Monitor** nearby plants daily for signs of spread
                """)

                st.markdown("### Additional Notes")
                st.info(treatment_info['suggestion'])

            st.markdown("---")
            st.markdown("### Export Diagnosis Report")
            try:
                pdf_bytes = generate_pdf(prediction, confidence, treatment_info, healthy)
                st.download_button(
                    label="Download Diagnosis Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"LeafScan_Report_{prediction.replace('___', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as pdf_err:
                st.error(f"PDF generation failed: {pdf_err}")

else:
    # =====================================================
    # LANDING / EXAMPLE IMAGES VIEW
    # =====================================================
    col_empty1, col_empty2, col_empty3 = st.columns([1, 2, 1])

    with col_empty2:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 40px;">
            <h3>Upload a leaf image or select an example below to begin analysis.</h3>
            <p>The AI model will classify the disease and provide a detailed treatment report.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("Symptom-Based Pre-Diagnostic Assistant (Optional)"):
        st.markdown("Select the symptoms observed on the plant leaf. The system will estimate the most probable disease matches.")

        col_sym1, col_sym2 = st.columns(2)
        with col_sym1:
            sym_yellow = st.checkbox("Yellow spots or discoloration patches")
            sym_brown = st.checkbox("Brown or black circular spots")
            sym_wilt = st.checkbox("Leaf wilting or drooping stems")
        with col_sym2:
            sym_webs = st.checkbox("Fine webbing on leaf undersides")
            sym_curl = st.checkbox("Curled, crinkled, or deformed leaf shape")
            sym_mold = st.checkbox("White, grey, or velvety powdery coating")

        selected_symptoms = []
        if sym_yellow: selected_symptoms.append("yellow")
        if sym_brown: selected_symptoms.append("brown")
        if sym_wilt: selected_symptoms.append("wilt")
        if sym_webs: selected_symptoms.append("webs")
        if sym_curl: selected_symptoms.append("curl")
        if sym_mold: selected_symptoms.append("mold")

        if selected_symptoms:
            st.markdown("##### Probable Disease Matches:")
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
                    st.write(f"**{dis_name}** ({match_percentage:.0f}% symptom match)")
                    st.progress(item[1])
            else:
                st.info("No disease matches found for the selected symptoms. Please upload a leaf photograph for AI-based analysis.")
        else:
            st.info("Select one or more symptoms above to view estimated disease matches.")

    st.markdown("---")
    st.subheader("Example Leaf Images")
    st.markdown("Select one of the reference images below to run an immediate AI diagnosis.")

    st.markdown("#### Healthy Specimens (4 Examples)")
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
                st.warning(f"Image not found: {info['file']}")

    st.markdown("#### Diseased Specimens (6 Examples)")
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
                st.warning(f"Image not found: {info['file']}")


# =====================================================
# PLANT PATHOLOGY CHATBOT
# =====================================================
st.markdown("---")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Welcome to the LeafScan Plant Pathology Assistant. "
                "I can provide information on plant diseases, treatments, soil conditions, and general crop management. "
                "Please enter your query below."
            )
        }
    ]

col_chat_title, col_chat_clear = st.columns([4, 1])
with col_chat_title:
    st.markdown("### Plant Pathology Assistant")
with col_chat_clear:
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to the LeafScan Plant Pathology Assistant. "
                    "I can provide information on plant diseases, treatments, soil conditions, and general crop management. "
                    "Please enter your query below."
                )
            }
        ]
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your question (e.g., 'What is Early Blight?' or 'How should tomatoes be irrigated?')"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Processing your query..."):
            response = get_chatbot_response(prompt, treatments)
        response_placeholder.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})


# =====================================================
# FOOTER
# =====================================================
st.markdown(f"""
<div class="footer-student">
    <p><b>LeafScan v2.5</b> | AI-Powered Plant Pathology Laboratory</p>
    <p>Built with TensorFlow, Keras, and Streamlit | Powered by MobileNetV2 Deep Learning</p>
    <p><b>Abdullah Mohammad Mushtaq | IQRA University</b></p>
</div>
""", unsafe_allow_html=True)
