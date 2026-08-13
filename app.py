import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Coffee Disease Prediction",
    page_icon="☕",
    layout="wide"
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
class_names = ["Health", "Leaf Rust", "Phoma"]

model_metrics = {
    "CNN": {
        "Accuracy": 0.7709,
        "Precision": 0.7902,
        "Recall": 0.7666,
        "F1 Score": 0.7572
    },
    "ResNet18": {
        "Accuracy": 0.9903,
        "Precision": 0.9908,
        "Recall": 0.9899,
        "F1 Score": 0.9902
    },
    "EfficientNet": {
        "Accuracy": 0.9942,
        "Precision": 0.9941,
        "Recall": 0.9942,
        "F1 Score": 0.9941
    },
    "MobileNet": {
        "Accuracy": 0.9942,
        "Precision": 0.9940,
        "Recall": 0.9944,
        "F1 Score": 0.9941
    }
}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model(model_name):

    if model_name == "CNN":

        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(224 * 224 * 3, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, len(class_names))
                )

            def forward(self, x):
                x = self.flatten(x)
                logits = self.linear_relu_stack(x)
                return logits

        model = NeuralNetwork()
        model_path = "models/best_CNN.pth"

    elif model_name == "ResNet18":

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features,
            len(class_names)
        )
        model_path = "models/best_ResNet18.pth"

    elif model_name == "EfficientNet":

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(class_names)
        )
        model_path = "models/best_EfficientNet.pth"

    elif model_name == "MobileNet":

        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            len(class_names)
        )
        model_path = "models/best_MobileNet.pth"

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# Header
st.title("☕ Coffee Disease Prediction")
st.write(
    "Upload a coffee leaf image to predict whether it is "
    "Healthy, affected by Leaf Rust, or affected by Phoma."
)

# Sidebar
st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "MobileNet",
        "EfficientNet",
        "ResNet18",
        "CNN"
        
    ]
)

# Load selected model
model = load_model(model_name)

# Upload image
uploaded_file = st.file_uploader(
    "Upload a coffee leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            image,
            use_container_width=True
        )

    # Prepare image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = predicted_class.item()
    confidence = confidence.item()

    probabilities = probabilities.squeeze().cpu().numpy()

    prediction = class_names[predicted_class]

    with col2:

        st.subheader("Prediction")

        st.success(prediction)

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.subheader("Class Probabilities")

        probability_data = pd.DataFrame({
            "Class": class_names,
            "Probability": probabilities * 100
        })

        probability_data["Probability"] = (
            probability_data["Probability"].round(2)
        )

        st.dataframe(
            probability_data,
            hide_index=True,
            use_container_width=True
        )

        st.bar_chart(
            probability_data.set_index("Class")
        )


# Model performance
st.divider()

st.subheader(f"{model_name} Performance")

metrics = model_metrics[model_name]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{metrics['Accuracy'] * 100:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{metrics['Precision'] * 100:.2f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{metrics['Recall'] * 100:.2f}%"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{metrics['F1 Score'] * 100:.2f}%"
    )


# Model information
st.divider()

st.subheader("Model Information")

st.write(f"**Selected model:** {model_name}")
st.write(f"**Classes:** {', '.join(class_names)}")
st.write(f"**Device:** {device}")
