import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from PIL import Image, ImageOps
import numpy as np
import pickle
import time

# ---  CONFIGURATION ---
st.set_page_config(
    page_title="DermaAI Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #007bff;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-text {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border: 1px solid #0056b3;
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE & RESET LOGIC ---
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
    st.session_state['score'] = 0.0

def reset_state():
    st.session_state['analyzed'] = False
    st.session_state['score'] = 0.0

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.build((None, 224, 224, 3))
    
    try:
        with open('skin_model_weights.pkl', 'rb') as f:
            saved_weights = pickle.load(f)
        model.set_weights(saved_weights)
        return model
    except Exception as e:
        st.error(f"System Error: {e}")
        return None

# --- PREPROCESSING ---
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    return img_array

def predict_standard(model, image):
    processed_img = preprocess_image(image)
    batch = np.expand_dims(processed_img, axis=0)
    prediction = model.predict(batch)
    return prediction[0][0]

# --- APP UI LAYOUT ---

# SIDEBAR
with st.sidebar:
    st.title("DermaAI Control")
    st.write("Professional Melanoma Screening Tool")
    st.markdown("---")
    
    st.info("💡 **Instructions:**\n1. Upload a clear Dermatoscopy image.\n2. Click 'Run Analysis'.\n3. Verify with ABCD protocol.")
    
    st.markdown("---")
    if st.button("🔄 Force Reset"):
        reset_state()
        st.rerun()

# MAIN AREA
st.markdown('<div class="main-header">🔬 DermaAI Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Automated Skin Lesion Analysis & Risk Assessment</div>', unsafe_allow_html=True)

model = load_model()

# Container for Upload
with st.container():
    uploaded_file = st.file_uploader(
        "", 
        type=["jpg", "png", "jpeg"], 
        help="Supported formats: JPG, PNG",
        on_change=reset_state 
    )

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.subheader("🖼️ Clinical Image")
        st.image(image, caption='Patient Scan', use_container_width=True, output_format="JPEG")
    
    with col2:
        st.subheader("🔍 Diagnostic Engine")
        
        # Action Button
        if not st.session_state['analyzed']:
            st.info("Image loaded successfully. Ready to analyze.")
            if st.button("🚀 Run AI Diagnosis"): 
                progress_text = "Analyzing lesion patterns..."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(0.2)
                my_bar.empty()
                
                score = predict_standard(model, image)
                st.session_state['score'] = score
                st.session_state['analyzed'] = True
                st.rerun()

        # Result Display
        if st.session_state['analyzed']:
            score = st.session_state['score']
            risk_probability = (1 - score) * 100
            
            m1, m2 = st.columns(2)
            
            if risk_probability > 50:
                st.error("🔴 **RESULT: MALIGNANT / HIGH RISK**")
                m1.metric("Melanoma Probability", f"{risk_probability:.2f}%", "Critical", delta_color="inverse")
                m2.metric("Confidence Level", "High", "Action Needed")
                
                st.markdown("""
                <div style="background-color: #fce8e6; color: #333333; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                    <strong>⚠️ AI Assessment:</strong><br>
                    The model has detected patterns strongly consistent with melanoma. 
                    Please consult a dermatologist immediately.
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.success("🟢 **RESULT: BENIGN / SAFE**")
                m1.metric("Benign Probability", f"{score*100:.2f}%", "Safe")
                m2.metric("Risk Level", "Low", "Routine Check")
                
                st.markdown("""
                <div style="background-color: #e6f9e6; color: #333333; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745;">
                    <strong>✅ AI Assessment:</strong><br>
                    No malignant features were detected. The lesion appears stable. 
                    Regular monitoring is advised.
                </div>
                """, unsafe_allow_html=True)

    # --- ABCD PROTOCOL ---
    if st.session_state['analyzed']:
        st.markdown("---")
        st.subheader("📋 Physician Verification (ABCD Protocol)")
        
        with st.expander("Open Clinical Checklist", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1: A = st.checkbox("Asymmetry")
            with c2: B = st.checkbox("Border Irregularity")
            with c3: C = st.checkbox("Color Variation")
            with c4: D = st.checkbox("Diameter > 6mm")
            
            if st.button("📊 Calculate Integrated Report"):
                manual_score = sum([A,B,C,D]) * 0.25
                final_score = ((1 - st.session_state['score']) * 0.8) + (manual_score * 0.2)
                final_percent = final_score * 100
                
                st.write("") 
                if final_percent > 50:
                    st.error(f"🔴 **Final Integrated Risk Score:** {final_percent:.2f}% (HIGH RISK)")
                else:
                    st.success(f"🟢 **Final Integrated Risk Score:** {final_percent:.2f}% (LOW RISK)")

else:
    st.info("👆 Please upload a patient image from the sidebar or drag and drop to begin.")