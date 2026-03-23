import streamlit as st
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from model import ConvLSTM

# =========================
# SETTINGS
# =========================

PATCH_SIZE = 128
device = "cpu"

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    model = ConvLSTM(input_dim=1)
    model.load_state_dict(torch.load("convlstm_insat3d.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# =========================
# HELPERS
# =========================

def preprocess(img):
    return img.astype("float32") / 1023.0

def crop_to_divisible(img, ps):
    H, W = img.shape
    return img[:(H//ps)*ps, :(W//ps)*ps]

def get_patch_coords(H, W, ps):
    coords = []
    for y in range(0, H - ps + 1, ps):
        for x in range(0, W - ps + 1, ps):
            coords.append((x, y))
    return coords

def reconstruct(patches, coords, H, W, ps):
    full = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for patch, (x, y) in zip(patches, coords):
        full[y:y+ps, x:x+ps] += patch
        count[y:y+ps, x:x+ps] += 1

    return full / (count + 1e-6)

# =========================
# TIMESTAMP EXTRACTION
# =========================

def extract_timestamp(filename):
    try:
        parts = filename.split("_")
        date_str = parts[1]   # 24MAY2024
        time_str = parts[2]   # 0030
        return datetime.strptime(date_str + time_str, "%d%b%Y%H%M")
    except:
        return None

def format_ts(ts):
    return ts.strftime("%d %b %Y %H:%M")

# =========================
# UI
# =========================

st.title("🛰️ INSAT-3D Nowcasting (ConvLSTM)")

st.write("Upload 4 sequential images (t-90, t-60, t-30, t)")

uploaded_files = st.file_uploader(
    "Upload 4 images", type=["tif"], accept_multiple_files=True
)

# =========================
# DISPLAY INPUTS
# =========================

if uploaded_files and len(uploaded_files) == 4:

    st.subheader("Input Sequence")

    cols = st.columns(4)

    timestamps = []

    for i, file in enumerate(uploaded_files):
        filename = file.name
        ts = extract_timestamp(filename)

        if ts:
            timestamps.append(ts)
            label = format_ts(ts)
        else:
            timestamps.append(None)
            label = "Unknown Time"

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        cols[i].image(img, caption=label, clamp=True)

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    if uploaded_files is None or len(uploaded_files) != 4:
        st.error("Please upload exactly 4 images")
    else:

        imgs = []
        timestamps = []

        for file in uploaded_files:
            filename = file.name
            ts = extract_timestamp(filename)
            timestamps.append(ts)

            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            img = preprocess(img)
            img = crop_to_divisible(img, PATCH_SIZE)

            imgs.append(img)

        H, W = imgs[0].shape
        coords = get_patch_coords(H, W, PATCH_SIZE)

        pred30_patches = []
        pred60_patches = []

        for (x, y) in coords:

            patch_seq = [img[y:y+PATCH_SIZE, x:x+PATCH_SIZE] for img in imgs]

            X = np.stack(patch_seq)
            X = torch.tensor(X).unsqueeze(1).unsqueeze(0)

            with torch.no_grad():
                pred = model(X, future=2)

            pred = pred.numpy()

            pred30_patches.append(pred[0, 0, 0])
            pred60_patches.append(pred[0, 1, 0])

        pred30 = reconstruct(pred30_patches, coords, H, W, PATCH_SIZE)
        pred60 = reconstruct(pred60_patches, coords, H, W, PATCH_SIZE)

        # =========================
        # COMPUTE FUTURE TIMESTAMPS
        # =========================

        last_ts = timestamps[-1]

        if last_ts:
            t30 = last_ts + timedelta(minutes=30)
            t60 = last_ts + timedelta(minutes=60)

            label30 = format_ts(t30)
            label60 = format_ts(t60)
        else:
            label30 = "t+30"
            label60 = "t+60"

        # =========================
        # DISPLAY RESULTS
        # =========================

        st.subheader("Predicted Outputs")

        col1, col2 = st.columns(2)

        with col1:
            st.image(pred30, caption=f"Prediction: {label30}", clamp=True)

        with col2:
            st.image(pred60, caption=f"Prediction: {label60}", clamp=True)