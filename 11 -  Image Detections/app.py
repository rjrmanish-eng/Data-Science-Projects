# app.py - Complete CIFAR-10 + CLIP Vision App
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from clip_helper import CLIPVision

# ---------- CIFAR-10 MODEL CLASS ----------
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ---------- LOAD CIFAR-10 MODEL ----------
@st.cache_resource
def load_cifar_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10_CNN().to(device)
    model_path = 'cifar10_model_epoch_20.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    else:
        st.warning(f"⚠️ CIFAR-10 model file '{model_path}' nahi mili. Sirf CLIP Vision chalega.")
        return None, device

# ---------- LOAD CLIP VISION ----------
@st.cache_resource
def load_clip_model():
    """CLIP model load karo - local, no API needed"""
    try:
        return CLIPVision()
    except Exception as e:
        st.error(f"❌ CLIP model load nahi ho paya: {str(e)}")
        st.info("💡 Install CLIP: pip install git+https://github.com/openai/CLIP.git")
        return None

# ---------- CLASSES ----------
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="AI Image Classifier", page_icon="🤖", layout="wide")
st.title("🤖 Dual-Mode Image Classifier")
st.write("Upload karo image - **CIFAR-10** + **CLIP Vision** dono try karega!")

# Load models
cifar_model, device = load_cifar_model()
clip_model = load_clip_model()

if clip_model is None:
    st.stop()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("CIFAR-10 Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    show_top = st.slider("Show Top N predictions", 1, 5, 3)
    
    st.divider()
    st.markdown("""
    ### 📋 About
    - **CIFAR-10**: 10 classes ki images
    - **CLIP Vision**: Kisi bhi image ko pehchanta hai
    - **100% Local**: Koi API nahi, free hai
    """)

CONFIDENCE_THRESHOLD = threshold

# Main content
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Image show karo
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📊 CIFAR-10 Result:")
        
        if cifar_model is not None:
            # CIFAR-10 prediction
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = cifar_model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                cifar_pred = output.argmax(1).item()
                cifar_confidence = probs[cifar_pred].item()
            
            if cifar_confidence >= CONFIDENCE_THRESHOLD:
                st.success(f"🎯 **Prediction: {classes[cifar_pred]}**")
                st.info(f"Confidence: {cifar_confidence*100:.2f}%")
            else:
                st.warning(f"⚠️ Low confidence ({cifar_confidence*100:.2f}%)")
                st.write(f"Best guess: {classes[cifar_pred]}")
            
            # Top N CIFAR-10 predictions
            st.write(f"### 📈 Top {show_top} CIFAR-10 Predictions:")
            top_probs, top_indices = torch.topk(probs, show_top)
            
            for i in range(show_top):
                idx = top_indices[i].item()
                prob = top_probs[i].item()
                st.progress(prob, text=f"{classes[idx]}: {prob*100:.2f}%")
        else:
            st.error("CIFAR-10 model not found")
    
    with col2:
        st.write("### 🤖 CLIP Vision (Local Model):")
        
        with st.spinner("Analyzing with CLIP..."):
            # Get object and description
            obj, desc = clip_model.analyze_image(image)
            
            # Show results
            if obj != "error":
                st.success(f"**Object:** {obj}")
                st.info(f"**Description:** {desc}")
            else:
                st.error(desc)
            
            # Show top predictions
            st.write(f"### 🎯 Top {show_top} CLIP Predictions:")
            top_predictions = clip_model.get_top_predictions(image, show_top)
            
            for cat, conf in top_predictions:
                if conf > 0:
                    st.progress(min(conf, 1.0), text=f"{cat}: {conf*100:.2f}%")
    
    # Full analysis
    with st.expander("🔍 Detailed Analysis", expanded=False):
        st.write("### Raw Model Outputs")
        st.json({
            "cifar10_available": cifar_model is not None,
            "clip_available": clip_model is not None,
            "threshold": CONFIDENCE_THRESHOLD,
            "image_size": f"{image.size[0]}x{image.size[1]}"
        })