import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os


st.set_page_config(
    page_title="coffeecare",
    page_icon="sucafina.svg",
    layout="wide"
)

st.sidebar.markdown(
    """
    <div style="
        font-size: 13px;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        line-height: 1.8;
        color: var(--text-color);
        text-align: justify;
    ">

    <b>TOOL OVERVIEW:</b><br><br>

    This tool uses deep learning to assess the health status of coffee leaves from uploaded images.
    It classifies leaves into three categories: <b>Healthy</b>, <b>Leaf Rust</b>, and <b>Phoma</b>.

    Users can select from multiple pre-trained deep learning models, including
    <b>CNN, ResNet18, EfficientNet</b>, and <b>MobileNet</b>, and review their performance metrics.

    For each uploaded image, the tool provides the <b>predicted class</b>,
    <b>confidence score</b>, and <b>class probabilities</b>, helping users understand
    both the prediction and the model's level of confidence.

    </div>
    """,
    unsafe_allow_html=True
)


#======================================================================================================================================

# MAIN TOOL PAGE SETTUP FOR THE STREAMLIT APP
#======================================================================================================================================
st.markdown("""
<style>
:root {
    --brand-color: #15767f;
    --brand-hover: #246b45;
    --brand-dark: #218838;
}

@media (prefers-color-scheme: dark) {
    :root {
        --brand-color: #4dd0e1;
        --brand-hover: #26c6da;
        --brand-dark: #1ba9b8;
    }
}

.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    margin-top: 0rem;
}

.fixed-header {
    position: absolute;
    top: 3rem;
    left: 0;       
    width: 100%;
    z-index: 9999;
    text-align: center;
    padding: 2px 2px 2px 2px;
    border-bottom: 2px solid var(--brand-color);
}

.fixed-header img {
    display: block;
    margin: 0 auto;
    width: 200px;  
    height: auto;
}

.fixed-header h1 {
    font-family: 'Poppins', sans-serif;
    color: var(--brand-color);
    font-size: 26px;
    margin: 0 0 0 0;
}

.content {
    margin-top: 270px;
}
            
.content-headings {
    font-family: 'Poppins', sans-serif;
    color: var(--brand-color);
    font-size: 24px;
    font-weight: bold;
}
            
@media (max-width: 1200px) {
    .fixed-header {
        top: 3rem;
        padding: 2px 2px 2px 2px;
        border-bottom-width: 2px;
    }

    .fixed-header img {
        width: 200px;
    }

    .fixed-header h1 {
        font-size: 26px;
        margin: 0 0 0 0;
    }

    .content {
        margin-top: 270px;
    }
            
    .content-headings {
        font-family: 'Poppins', sans-serif;
        color: var(--brand-color);
        font-size: 24px;
        font-weight: bold;
    }
}

            
@media (max-width: 900px) {
    .fixed-header {
        top: 3.2rem;
        padding: 2px 2px 2px 2px;
        border-bottom-width: 1px;
    }

    .fixed-header img {
        width: 150px;
    }

    .fixed-header h1 {
        font-size: 18px;
        margin: 1px 0 0 0;
    }

    .content {
        margin-top: 225px;
    }
            
    .content-headings {
        font-family: 'Poppins', sans-serif;
        color: var(--brand-color);
        font-size: 15px;
    }
}            


@media (max-width: 600px) {
    .fixed-header {
        top: 3.5rem;
        padding: 0px 0px 0px 0px;
    }

    .fixed-header img {
        width: 100px;
    }

    .fixed-header h1 {
        font-size: 13px;
    }

    .content {
        margin-top: 100px;
    }
            
    .content-headings {
        font-family: 'Poppins', sans-serif;
        color: var(--brand-color);
        font-size: 12px;
    }
}
</style>

<div class="fixed-header">
    <img src="https://maprinitiative.org/wp-content/uploads/2026/08/Sucafina-Logo.jpg" alt="Sucafina Logo">
    <h1>Coffee Disease Prediction Tool</h1>
</div>

<div class="content"></div>
""", unsafe_allow_html=True)






#======================================================================================================================================
# DATA LOADING AND PROCESSING COLUMN 
#======================================================================================================================================

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
class_names = ["Health", "Leaf Rust", "Phoma"]

model_metrics = {
    "CNN": {
        "Accuracy": 0.9049,
        "Precision": 0.9183,
        "Recall": 0.9072,
        "F1 Score": 0.9043
    },
    "ResNet18": {
        "Accuracy": 0.9922,
        "Precision": 0.9924,
        "Recall": 0.9921,
        "F1 Score": 0.9922
    },
    "EfficientNet": {
        "Accuracy": 0.9903,
        "Precision": 0.9901,
        "Recall": 0.9904,
        "F1 Score": 0.9902
    },
    "MobileNet": {
        "Accuracy": 0.9961,
        "Precision": 0.9961,
        "Recall": 0.9961,
        "F1 Score": 0.9961
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

        class CNN(nn.Module):
            def __init__(self, num_classes=3):
                super().__init__()

                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

                self.classifier = nn.Sequential(
                    nn.Flatten(),

                    nn.Linear(128 * 28 * 28, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),

                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x


        model = CNN(num_classes=3)
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


st.markdown(
    """
    <p style="font-size: 24px; font-weight: 500;">
        Upload a coffee leaf image to predict whether it is
        Healthy or affected by Leaf Rust, or Phoma.
    </p>
    """,
    unsafe_allow_html=True
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
st.markdown(
    """
    <hr style="
        border: none;
        border-top: 1px solid #15767f;
        margin: 30px 0;
    ">
    """,
    unsafe_allow_html=True
)

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

st.markdown(
    """
    <hr style="
        border: none;
        border-top: 1px solid #15767f;
        margin: 30px 0;
    ">
    """,
    unsafe_allow_html=True
)
