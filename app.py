import streamlit as st

# =========================
# SAFE IMPORTS (CRITICAL)
# =========================

st.title("🛰️ INSAT-3D Nowcasting (ConvLSTM)")

try:
    import torch
    import numpy as np
    from PIL import Image
    from datetime import datetime, timedelta
    from model import ConvLSTM

    st.success("✅ Imports successful")

except Exception as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# =========================
# SETTINGS
# =========================

PATCH_SIZE = 128
device = "cpu"

# =========================
# MODEL LOADING (SAFE)
# =========================

@st.cache_resource
def load_model():
    try:
        model = ConvLSTM(input_dim=1)
        model.load_state_dict(torch.load("convlstm_insat3d.pth", map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        return None

model = load_model()

# =========================
# HELPERS
# =========================

def preprocess(img):
    return img.astype("float32") / 1023.0


def crop_to_divisible(img, ps):
    H, W = img.shape
    return img[:(H // ps) * ps, :(W // ps) * ps]


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
# TIMESTAMP FUNCTIONS
# =========================

def extract_timestamp(filename):
    try:
        parts = filename.split("_")
        date_str = parts[1]
        time_str = parts[2]
        return datetime.strptime(date_str + time_str, "%d%b%Y%H%M")
    except:
        return None


def format_ts(ts):
    return ts.strftime("%d %b %Y %H:%M")


# =========================
# UI INPUT
# =========================

st.write("Upload 4 sequential INSAT-3D images (t-90, t-60, t-30, t)")

uploaded_files = st.file_uploader(
    "Upload 4 .tif images", type=["tif"], accept_multiple_files=True
)

# =========================
# DISPLAY INPUT IMAGES
# =========================

if uploaded_files and len(uploaded_files) == 4:

    st.subheader("Input Sequence")

    cols = st.columns(4)

    for i, file in enumerate(uploaded_files):
        try:
            ts = extract_timestamp(file.name)
            label = format_ts(ts) if ts else "Unknown Time"

            file.seek(0)
            img = np.array(Image.open(file))

            cols[i].image(img, caption=label, clamp=True)

        except Exception as e:
            cols[i].error(f"Error loading image: {e}")

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    if uploaded_files is None or len(uploaded_files) != 4:
        st.error("❌ Please upload exactly 4 images")

    elif model is None:
        st.error("❌ Model not loaded")

    else:
        try:
            imgs = []
            timestamps = []

            # Read images
            for file in uploaded_files:
                ts = extract_timestamp(file.name)
                timestamps.append(ts)

                file.seek(0)
                img = np.array(Image.open(file))

                img = preprocess(img)
                img = crop_to_divisible(img, PATCH_SIZE)

                imgs.append(img)

            H, W = imgs[0].shape
            coords = get_patch_coords(H, W, PATCH_SIZE)

            pred30_patches = []
            pred60_patches = []

            # Patch-wise prediction
            for (x, y) in coords:

                patch_seq = [img[y:y+PATCH_SIZE, x:x+PATCH_SIZE] for img in imgs]

                X = np.stack(patch_seq)
                X = torch.tensor(X).unsqueeze(1).unsqueeze(0)

                with torch.no_grad():
                    pred = model(X, future=2)

                pred = pred.numpy()

                pred30_patches.append(pred[0, 0, 0])
                pred60_patches.append(pred[0, 1, 0])

            # Reconstruct full images
            pred30 = reconstruct(pred30_patches, coords, H, W, PATCH_SIZE)
            pred60 = reconstruct(pred60_patches, coords, H, W, PATCH_SIZE)

            # Compute timestamps
            last_ts = timestamps[-1]

            if last_ts:
                t30 = last_ts + timedelta(minutes=30)
                t60 = last_ts + timedelta(minutes=60)

                label30 = format_ts(t30)
                label60 = format_ts(t60)
            else:
                label30 = "t+30"
                label60 = "t+60"

            # Display results
            st.subheader("Predicted Outputs")

            col1, col2 = st.columns(2)

            with col1:
                st.image(pred30, caption=f"Prediction: {label30}", clamp=True)

            with col2:
                st.image(pred60, caption=f"Prediction: {label60}", clamp=True)

        except Exception as e:
            st.error(f"❌ Runtime Error: {e}")