# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ====== Model Architecture (same as training) ======
LATENT_DIM = 20
LABEL_DIM = 10
IMAGE_SIZE = 28 * 28

def get_onehot(labels):
    return torch.eye(LABEL_DIM)[labels]

class CVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(IMAGE_SIZE + LABEL_DIM, 400)
        self.fc_mu = torch.nn.Linear(400, LATENT_DIM)
        self.fc_logvar = torch.nn.Linear(400, LATENT_DIM)
        self.fc3 = torch.nn.Linear(LATENT_DIM + LABEL_DIM, 400)
        self.fc4 = torch.nn.Linear(400, IMAGE_SIZE)

    def encode(self, x, y):
        h = F.relu(self.fc1(torch.cat([x, y], 1)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], 1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# ====== Load Model ======
@st.cache_resource
def load_model():
    model = CVAE()
    model.load_state_dict(torch.load("weights/cvae_mnist.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ====== Streamlit UI ======
st.set_page_config(page_title="MNIST Digit Generator", layout="centered")
st.title("🖋️ Generate Handwritten Digits")
digit = st.selectbox("Select a digit (0-9)", list(range(10)))
if st.button("Generate 5 Images"):
    y = torch.tensor([digit] * 5)
    y_onehot = get_onehot(y)
    z = torch.randn(5, LATENT_DIM)
    generated = model.decode(z, y_onehot).detach().numpy().reshape(-1, 28, 28)

    cols = st.columns(5)
    for i in range(5):
        img = (generated[i] * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        cols[i].image(pil_img.resize((112, 112), Image.NEAREST), caption=f"{digit}")
