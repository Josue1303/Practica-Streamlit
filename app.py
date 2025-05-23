import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Parámetros
NUM_CLASSES = 23
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "abyssinian",
    "american shorthair",
    "beagle",
    "boxer",
    "bulldog",
    "chihuahua",
    "corgi",
    "dachshund",
    "german shepherd",
    "golden retriever",
    "husky",
    "labrador",
    "maine coon",
    "mumbai cat",
    "persian cat",
    "pomeranian",
    "pug",
    "ragdoll cat",
    "rottwiler",
    "shiba inu",
    "siamese cat",
    "sphynx",
    "yorkshire terrier"
]

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Cargar modelo
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load("modelo_mascotas.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# Interfaz de Streamlit
st.title("Clasificador de Razas de Mascotas 🐶🐱")
st.write("Sube una imagen de una mascota para predecir su raza.")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesar y predecir
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = CLASS_NAMES[predicted.item()]
        st.success(f"Raza Predicha: **{pred_class}**")
