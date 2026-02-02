import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

# --- 1. Configuration & Constants ---
TARGET_CLASSES = ['Anthracnose', 'Bacterial Wilt', 'Downy Mildew', 'Fresh Leaf', 'Gummy Stem Blight']
IMAGE_SIZE = 299 
MODELS_DIR = "models"  # Directory where .pth files are stored
SAMPLES_DIR = "samples" # Directory where sample images are stored

# Updated keys to show architecture names in the UI
MODEL_CONFIGS = {
    'ResNet50': 'base',
    'InceptionV3': 'base',
    'EfficientNetB0': 'base',
    'MobileNet': 'base',
    'Hybrid-1 (ResNet50 + InceptionV3)': ('ResNet50', 'InceptionV3'),
    'Hybrid-2 (ResNet50 + EfficientNetB0)': ('ResNet50', 'EfficientNetB0'),
    'Hybrid-3 (MobileNet + EfficientNetB0)': ('MobileNet', 'EfficientNetB0')
}

# Mapping specific UI names to expected filename prefixes if they differ
# Or simply rename your .pth files to match the keys in MODEL_CONFIGS exactly.
# For simplicity, this code assumes your file is named like the key (e.g., "ResNet50.pth")
# or you can rename the file manually.

# --- 2. Model Definitions ---
class HybridModel(nn.Module):
    def __init__(self, model_a_name, model_b_name, num_classes):
        super(HybridModel, self).__init__()
        self.model_a, feat_a = self._get_backbone(model_a_name)
        self.model_b, feat_b = self._get_backbone(model_b_name)
        self.out_features = feat_a + feat_b
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def _get_backbone(self, name):
        if name == 'ResNet50':
            m = models.resnet50(weights=None)
            num_ftrs = m.fc.in_features
            m.fc = nn.Identity()
            return m, num_ftrs
        elif name == 'InceptionV3':
            # Note: aux_logits=False is critical here
            m = models.inception_v3(weights=None, aux_logits=False, init_weights=False)
            num_ftrs = m.fc.in_features
            m.fc = nn.Identity()
            return m, num_ftrs
        elif name == 'EfficientNetB0':
            m = models.efficientnet_b0(weights=None)
            num_ftrs = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, num_ftrs
        elif name == 'MobileNet':
            m = models.mobilenet_v2(weights=None)
            num_ftrs = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, num_ftrs
        return None, 0

    def forward(self, x):
        # InceptionV3 expects (299, 299). If inputs differ, we might need resizing here,
        # but since we set global IMAGE_SIZE = 299, it fits both.
        out1 = torch.flatten(self.model_a(x), 1)
        out2 = torch.flatten(self.model_b(x), 1)
        combined = torch.cat((out1, out2), dim=1)
        return self.classifier(combined)

def get_base_model(model_name, num_classes):
    if model_name == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'InceptionV3':
        model = models.inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'MobileNet':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# --- 3. UI Styling & Layout ---
st.set_page_config(page_title="Cucumber AI Doctor", layout="wide", page_icon="🥒")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white; border: none; }
    .stButton>button:hover { background-color: #1b5e20; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🥒 Cucumber Disease Diagnostic Lab")

with st.expander("📖 User Guide: How to use this application", expanded=False):
    st.markdown("""
    **Welcome to the AI Diagnostic Lab.** This tool uses Deep Learning to identify diseases in cucumber leaves.
    
    ### 👣 Step-by-Step Instructions:
    
    1.  **Select Architecture (Sidebar):** * Choose a specific AI model from the sidebar menu on the left.
        * *Note:* If the system cannot find the model file locally, you will be prompted to upload the `.pth` weight file.
    
    2.  **Input Image (Section 1):**
        * **Option A:** Upload a picture of a cucumber leaf from your computer (JPG/PNG).
        * **Option B:** Select a pre-loaded test image from the "Sample Images" tab.
    
    3.  **Run Diagnosis:**
        * Once an image is visible, click the green **"🚀 Analyze Leaf"** button in Section 2.
    
    4.  **View Results:**
        * The AI will display the detected disease, a confidence chart, and a recommended **Action Plan** for treatment.
    """)
# ----------------------------------

# --- 4. Sidebar: Model Selection ---
st.sidebar.header("🛠️ Lab Configuration")
selected_model_key = st.sidebar.selectbox("Select Architecture", list(MODEL_CONFIGS.keys()))

# LOGIC: Check if file exists in 'models/' folder, otherwise ask for upload
# We create a simple filename by stripping special chars or you can map manually
safe_filename = selected_model_key.split(" (")[0] + ".pth" # e.g., "Hybrid-1.pth"
local_model_path = os.path.join(MODELS_DIR, safe_filename)

model_source = None

if os.path.exists(local_model_path):
    st.sidebar.success(f"✅ Found built-in weights: {safe_filename}")
    model_source = local_model_path
else:
    st.sidebar.warning(f"⚠️ '{safe_filename}' not found in '{MODELS_DIR}/'.")
    uploaded_weights = st.sidebar.file_uploader(f"Upload weights for {selected_model_key}", type=["pth"])
    if uploaded_weights:
        model_source = uploaded_weights
        st.sidebar.success("✅ Weights uploaded!")

# --- 5. Helper Functions ---
@st.cache_resource
def load_model_instance(model_key, source_path):
    config = MODEL_CONFIGS[model_key]
    
    # 1. Initialize Architecture
    if config == 'base':
        model = get_base_model(model_key, len(TARGET_CLASSES))
    else:
        # Tuple: ('ResNet50', 'InceptionV3')
        model = HybridModel(config[0], config[1], len(TARGET_CLASSES))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Weights with Sanitization (Fixes Hybrid-1 Error)
    try:
        if isinstance(source_path, str):
            state_dict = torch.load(source_path, map_location=device)
        else:
            state_dict = torch.load(source_path, map_location=device)
            
        # FIX FOR HYBRID-1 / INCEPTION ERRORS:
        # Inception weights often contain "AuxLogits" keys. If we initialized with aux_logits=False,
        # these keys cause an error. We simply filter them out.
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove AuxLogits keys
            if "AuxLogits" in k:
                continue
            new_state_dict[k] = v
            
        # Load state dict with strict=False to allow flexibility (helps with version mismatches)
        model.load_state_dict(new_state_dict, strict=False)
        
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        return None, None

    model.to(device)
    model.eval()
    return model, device

# --- 6. Main Content: Image Input ---
col_input, col_results = st.columns([1, 1])

final_image = None

with col_input:
    st.subheader("1. Select Specimen")
    
    tab1, tab2 = st.tabs(["📤 Upload Image", "🖼️ Sample Images"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            final_image = Image.open(uploaded_file).convert('RGB')
            
    with tab2:
        # Load samples from folder
        if not os.path.exists(SAMPLES_DIR):
            os.makedirs(SAMPLES_DIR) # Create if doesn't exist to prevent error
            
        sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_files:
            selected_sample = st.selectbox("Choose a test image", sample_files)
            if selected_sample:
                image_path = os.path.join(SAMPLES_DIR, selected_sample)
                final_image = Image.open(image_path).convert('RGB')
        else:
            st.info(f"No images found in '{SAMPLES_DIR}/' folder.")

    if final_image:
        st.image(final_image, caption="Specimen for Analysis", use_container_width=True)

# --- 7. Prediction & Analysis ---
with col_results:
    st.subheader("2. Diagnostic Report")
    
    if final_image and model_source:
        if st.button("🚀 Analyze Leaf"):
            with st.spinner('Initializing Neural Networks...'):
                model, device = load_model_instance(selected_model_key, model_source)
                
                if model:
                    # Preprocess
                    transform = transforms.Compose([
                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(final_image).unsqueeze(0).to(device)
                    
                    # Inference
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                        conf, pred = torch.max(probs, 0)
                        label = TARGET_CLASSES[pred.item()]
                    
                    # --- RESULTS UI ---
                    st.success("Analysis Successfully Completed")
                    
                    # Visual Card
                    bg_color = "#d4edda" if label == 'Fresh Leaf' else "#f8d7da"
                    text_color = "#155724" if label == 'Fresh Leaf' else "#721c24"
                    border_color = "#c3e6cb" if label == 'Fresh Leaf' else "#f5c6cb"

                    st.markdown(f"""
                        <div style="
                            background-color: {bg_color};
                            padding: 20px;
                            border-radius: 10px;
                            border: 2px solid {border_color};
                            text-align: center;
                            margin-bottom: 20px;">
                            <h4 style="color: {text_color}; margin: 0;">DETECTED CONDITION</h4>
                            <h1 style="color: {text_color}; margin: 10px 0;">{label}</h1>
                            <h3 style="color: {text_color}; opacity: 0.8; margin: 0;">Confidence: {conf.item():.2%}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Chart
                    st.write("### Model Confidence Levels")
                    chart_df = pd.DataFrame({
                        'Condition': TARGET_CLASSES,
                        'Probability': [float(p) for p in probs]
                    }).set_index('Condition')
                    st.bar_chart(chart_df)
                    
                    # Action Plan
                    with st.expander("📋 View Action Plan", expanded=True):
                        if label == 'Fresh Leaf':
                            st.info("Plant is healthy. No action required.")
                        elif label == 'Anthracnose':
                            st.write("• **Action:** Remove infected leaves. Apply copper-based fungicide.")
                        elif label == 'Bacterial Wilt':
                            st.write("• **Action:** Control cucumber beetles. Remove infected plants immediately.")
                        elif label == 'Downy Mildew':
                            st.write("• **Action:** Improve air circulation. Apply fungicide specifically for Downy Mildew.")
                        elif label == 'Gummy Stem Blight':
                            st.write("• **Action:** Rotate crops. Apply protectant fungicides.")
    
    elif not final_image:
        st.info("👈 Please select or upload an image to begin.")
    elif not model_source:
        st.error(f"Please upload weights for **{selected_model_key}** or add them to the 'models/' folder.")

st.markdown("---")
st.caption("AI Model trained on Cucumber Disease Recognition Dataset.")