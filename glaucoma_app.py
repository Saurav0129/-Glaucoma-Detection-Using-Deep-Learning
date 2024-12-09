import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import uuid
import matplotlib.gridspec as gridspec

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_extractor = model[0]  
        self.target_layer = self.feature_extractor[-3]
        self.gradients = None
        self.activations = None
 
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, x, class_idx=None):
        self.model.eval()
        
     
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1

        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        cam = torch.sum(weights[:, :, None, None] * self.activations, dim=1)
        cam = F.relu(cam)

        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.detach().cpu().numpy()

class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.feature_extractor = model[0]
        self.target_layer = self.feature_extractor[-3] 
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        grad = self.gradients[0]
        activation = self.activations[0]
        alpha_num = grad.pow(2)
        alpha_denom = 2.0 * grad.pow(2)
        alpha_denom += torch.sum(activation * grad.pow(3), dim=(1, 2), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-7)
 
        weights = torch.sum(alpha * F.relu(grad), dim=(1, 2))
        
        cam = torch.sum(weights[:, None, None] * activation, dim=0)
        cam = F.relu(cam)
        
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.detach().cpu().numpy()

def visualize_gradcam(model, image_tensor, label_mapping, true_label=None):
    # Initialize GradCAM and GradCAM++
    gradcam = GradCAM(model)
    gradcam_pp = GradCAMPlusPlus(model)
    
    # Get model prediction and probability
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        probability = probs[0][pred_idx].item()
    
    gradcam_map = gradcam(image_tensor)
    gradcam_pp_map = gradcam_pp(image_tensor)
    
    image = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    image = (image - image.min()) / (image.max() - image.min())
    
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])  
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.imshow(cv2.resize(gradcam_map[0], (image.shape[1], image.shape[0])), 
           cmap='jet', alpha=0.5)
    ax0.set_title('GradCAM')
    ax0.axis('off')

    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(image)
    ax1.imshow(cv2.resize(gradcam_pp_map, (image.shape[1], image.shape[0])), 
           cmap='jet', alpha=0.5)
    ax1.set_title('GradCAM++')
    ax1.axis('off')

    return fig


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_image = cv2.merge((cl, a, b))
    
    clahe_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    median_filtered_image = cv2.medianBlur(clahe_image, 5)
    
    image_rgb = cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2RGB)
  
    image_pil = Image.fromarray(image_rgb)
    
    return image_pil

def prepare_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0) 
    return image_tensor

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

def build_mobilenetv3_model(num_classes=2):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-5:].parameters():  
        param.requires_grad = True

    class SpatialSoftAttention(nn.Module):
        def __init__(self, in_channels):
            super(SpatialSoftAttention, self).__init__()
            self.max_pool = nn.MaxPool2d(kernel_size=1, stride=1)
            self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
            self.attn_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1)
        
        def forward(self, x):
            maxp_x = self.max_pool(x)
            avgp_x = self.avg_pool(x)
            spat_attn = self.attn_conv(torch.cat([maxp_x, avgp_x], dim=1))
            spat_attn = torch.sigmoid(spat_attn)
            return torch.mul(spat_attn, x)

    
    in_channels = model.features[-1][0].out_channels 
    modules = list(model.features)  
    modules.append(SpatialSoftAttention(in_channels)) 
    modules.append(nn.AdaptiveAvgPool2d((1, 1)))  

    
    classifier_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_channels, 1024), nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(1024, 512), nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
   
    modified_model = nn.Sequential(
        nn.Sequential(*modules),  
        classifier_head           
    )
    
    return modified_model  


def load_model(model_path):
    """Load the pre-trained model"""
    model = build_mobilenetv3_model(num_classes=2)  
    best_model_state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(best_model_state)
    model.to('cpu')
    model.eval()
    return model

def predict_image(model, image_tensor):
    """Make prediction on an image"""
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        probability = probs[0][pred_idx].item()
    return pred_idx, probability

os.makedirs('temp', exist_ok=True)

def main():

    st.set_page_config(page_title="Glaucoma Detection Tool", layout="wide")
    st.markdown(
        """
        <style>
            .title {
                font-size: 2.5rem;
                font-weight: bold;
                text-align: center;
                color: #2c3e50;
            }
            .sidebar-header {
                font-size: 1.2rem;
                font-weight: bold;
                color: #16a085;
            }
            .result-positive {
                background-color: #e74c3c;
                color: white;
                padding: 10px;
                border-radius: 10px;
                font-size: 1.5rem;
                text-align: center;
            }
            .result-negative {
                background-color: #27ae60;
                color: white;
                padding: 10px;
                border-radius: 10px;
                font-size: 1.5rem;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="title">Glaucoma Detection from Fundus Images</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-header">Upload Fundus Image</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    label_mapping = {0: "No Glaucoma", 1: "Glaucoma Detected"}
    
    if uploaded_file is not None:
        unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        temp_path = os.path.join("temp", unique_filename)
   
 
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
   
        preprocessed_image = preprocess_image(temp_path)
   
        os.unlink(temp_path)

        with col1:
            st.write("### Original Image")
            st.image(preprocessed_image, use_container_width=True)

        image_tensor = prepare_image(preprocessed_image)

        MODEL_PATH = r"C:\Users\skbis\Downloads\best_model.pth"
        
        model = load_model(MODEL_PATH)
        prediction, probability = predict_image(model, image_tensor)

        with col2:
            st.subheader("Results")
            if prediction == 1:
                st.markdown('<div class="result-positive">Glaucoma Detected</div>', unsafe_allow_html=True)
                st.write("Based on the prediction, there may be signs of glaucoma. However, this result is not a diagnosis.")
                st.write("It is highly recommended to schedule an appointment with an ophthalmologist for further evaluation and diagnosis.")
            else:
                st.markdown('<div class="result-negative">No Glaucoma</div>', unsafe_allow_html=True)
                st.write("Your results suggest that no glaucoma is detected. However, please continue regular eye check-ups to maintain eye health.")

            st.write(f"**Probability:** {probability:.2%}")
            st.write("These results are not a definitive diagnosis. Please consult an ophthalmologist for a comprehensive eye examination and proper medical advice.")
          
            st.write("You can find a list of qualified eye care professionals in your area here: [Find an Eye Care Professional](https://www.practo.com/doctors?utm_source=corecard&utm_medium=referral&utm_campaign=practohomepage)")  
            st.write("For more information on glaucoma and its symptoms, check trusted health resources such as [Trusted Health Resource](https://www.medicalnewstoday.com/articles/9710).")


        st.write("### Interpretation: GradCAM Visualization")
        grad_fig = visualize_gradcam(model, image_tensor, label_mapping)
        st.pyplot(grad_fig)
    
    else:
        st.info("Please upload a fundus image in the sidebar.")

if __name__ == "__main__":
    main()