import streamlit as st
import torch
from PIL import Image
import numpy as np
import pandas as pd
import faiss
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Visual Product Search", layout="wide")
st.title("👕 Visual Product Search Engine")
st.markdown("Upload an image, select a clothing region, and find matching products!")

# --- 1. LOAD MODELS (CACHED) ---
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load YOLO (Replace 'best_yolo.pt' with your actual weights file)
    yolo_model = YOLO('best_yolo.pt') 
    
    # 2. Load BLIP-2
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    
    # 3. Load CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    return yolo_model, blip_processor, blip_model, clip_model, clip_processor, device

@st.cache_resource
def load_index_and_data():
    # Load your FAISS index and the CSV mapping index rows to actual image paths
    index = faiss.read_index("product_index.index")
    metadata = pd.read_csv("product_metadata.csv") # Must contain an 'image_path' column
    return index, metadata

with st.spinner("Loading AI Models... This may take a minute on first run."):
    yolo_model, blip_processor, blip_model, clip_model, clip_processor, device = load_models()
    index, metadata = load_index_and_data()

# --- STATE MANAGEMENT ---
if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None
if 'crop_confirmed' not in st.session_state:
    st.session_state.crop_confirmed = False

# --- 2. SIDEBAR & UPLOAD ---
st.sidebar.header("Search Settings")
target_region = st.sidebar.radio("Select Target Region:", ["Upper Body", "Lower Body", "Full Body"])
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Define which YOLO classes belong to which region (Adjust indices based on your dataset!)
# Example: 0: short-sleeve, 1: long-sleeve, 2: skirt, 3: shorts, 4: trousers
REGION_MAP = {
    "Upper Body": [0, 1], 
    "Lower Body": [2, 3, 4],
    "Full Body": [0, 1, 2, 3, 4]
}

# --- 3. MAIN LOGIC ---
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)

    # YOLO DETECTION & CROPPING
    if st.session_state.cropped_img is None:
        if st.button("Detect & Crop"):
            results = yolo_model(original_image)
            boxes = results[0].boxes
            
            best_box = None
            max_conf = 0.0
            
            # Find the highest confidence box matching the requested region
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id in REGION_MAP[target_region] and conf > max_conf:
                    max_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()
            
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box)
                st.session_state.cropped_img = original_image.crop((x1, y1, x2, y2))
                st.rerun() # Refresh to show crop
            else:
                st.error(f"No {target_region} apparel detected. Try a different image.")

    # CROP CONFIRMATION
    if st.session_state.cropped_img is not None and not st.session_state.crop_confirmed:
        with col2:
            st.subheader("Detected Crop")
            st.image(st.session_state.cropped_img, width=250)
            
            st.warning("Please confirm the crop to proceed to search.")
            c1, c2 = st.columns(2)
            if c1.button("✅ Confirm Crop"):
                st.session_state.crop_confirmed = True
                st.rerun()
            if c2.button("🔄 Re-crop (Reset)"):
                st.session_state.cropped_img = None
                st.rerun()

    # RETRIEVAL (CONFIG B)
    if st.session_state.crop_confirmed:
        st.divider()
        st.subheader("Retrieving Similar Products...")
        
        with st.spinner("Generating Semantic Caption (BLIP-2)..."):
            crop = st.session_state.cropped_img
            
            # 1. BLIP-2 Caption
            inputs = blip_processor(crop, return_tensors="pt").to(device, torch.float16 if device=="cuda" else torch.float32)
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            st.info(f"**Generated Caption:** {caption}")
            
        with st.spinner("Fusing Multimodal Embeddings (CLIP)..."):
            # 2. CLIP Text Embedding
            text_inputs = clip_processor(text=[caption], return_tensors="pt", padding=True).to(device)
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            # 3. CLIP Image Embedding
            image_inputs = clip_processor(images=crop, return_tensors="pt").to(device)
            image_features = clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # 4. FUSION (Config B: alpha = 0.5)
            fused_embedding = (0.5 * image_features) + (0.5 * text_features)
            fused_embedding = fused_embedding / fused_embedding.norm(p=2, dim=-1, keepdim=True)
            fused_embedding = fused_embedding.cpu().detach().numpy().astype('float32')
            
        with st.spinner("Searching Catalog..."):
            # 5. FAISS Search
            K = 5
            distances, indices = index.search(fused_embedding, K)
            
            st.subheader("Top Matches")
            res_cols = st.columns(K)
            for idx, col in zip(indices[0], res_cols):
                # Fetch image path from metadata
                img_path = metadata.iloc[idx]['image_path'] 
                matched_img = Image.open(img_path)
                with col:
                    st.image(matched_img, use_column_width=True)
                    
        if st.button("Start New Search"):
            st.session_state.cropped_img = None
            st.session_state.crop_confirmed = False
            st.rerun()
