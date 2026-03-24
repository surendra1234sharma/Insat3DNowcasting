import streamlit as st

st.title("🛰️ INSAT-3D Nowcasting (ConvLSTM)")

# =========================
# IMPORTS
# =========================

try:
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    from model import ConvLSTM
except Exception as e:
    st.error(f"Import Error: {e}")
    st.stop()

PATCH_SIZE = 128
device = "cpu"

# =========================
# MODEL LOADING
# =========================

@st.cache_resource
def load_model():
    loaded = torch.load("convlstm_insat3d.pth", map_location=device)

    if isinstance(loaded, torch.nn.Module):
        model = loaded
    else:
        model = ConvLSTM()
        model.load_state_dict(loaded)

    model.eval()
    return model

model = load_model()

# =========================
# DISPLAY HELPER (IMPORTANT)
# =========================

def display_image(img, title=""):
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)

    fig, ax = plt.subplots()
    ax.imshow(img_norm, cmap="jet")
    ax.set_title(title)
    ax.axis("off")

    return fig

# =========================
# BASIC HELPERS
# =========================

def preprocess(img):
    return img.astype("float32") / 1023.0

def crop_to_divisible(img, ps):
    H, W = img.shape
    return img[:(H//ps)*ps, :(W//ps)*ps]

def get_patch_coords(H, W, ps):
    return [(x, y) for y in range(0, H-ps+1, ps) for x in range(0, W-ps+1, ps)]

def reconstruct(patches, coords, H, W, ps):
    full = np.zeros((H, W))
    count = np.zeros((H, W))

    for patch, (x, y) in zip(patches, coords):
        full[y:y+ps, x:x+ps] += patch
        count[y:y+ps, x:x+ps] += 1

    return full / (count + 1e-6)

def extract_timestamp(filename):
    try:
        parts = filename.split("_")
        return datetime.strptime(parts[1] + parts[2], "%d%b%Y%H%M")
    except:
        return None

def format_ts(ts):
    return ts.strftime("%d %b %Y %H:%M")

# =========================
# INPUT
# =========================

st.write("Upload 4 sequential images")

uploaded_files = st.file_uploader(
    "Upload (t-90, t-60, t-30, t)", type=["tif"], accept_multiple_files=True
)

# =========================
# SHOW INPUTS (FIXED)
# =========================

if uploaded_files and len(uploaded_files) == 4:

    st.subheader("Input Images")

    cols = st.columns(4)

    for i, file in enumerate(uploaded_files):
        file.seek(0)
        img = np.array(Image.open(file))

        ts = extract_timestamp(file.name)
        label = format_ts(ts) if ts else "Unknown"

        fig = display_image(img, label)
        cols[i].pyplot(fig)

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    imgs = []
    timestamps = []

    for file in uploaded_files:
        file.seek(0)
        img = np.array(Image.open(file))

        ts = extract_timestamp(file.name)
        timestamps.append(ts)

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
            pred = model(X, future=2).numpy()

        pred30_patches.append(pred[0, 0, 0])
        pred60_patches.append(pred[0, 1, 0])

    pred30 = reconstruct(pred30_patches, coords, H, W, PATCH_SIZE)
    pred60 = reconstruct(pred60_patches, coords, H, W, PATCH_SIZE)

    last_ts = timestamps[-1]

    if last_ts:
        t30 = format_ts(last_ts + timedelta(minutes=30))
        t60 = format_ts(last_ts + timedelta(minutes=60))
    else:
        t30, t60 = "t+30", "t+60"

    st.subheader("Predictions")

    col1, col2 = st.columns(2)

    col1.pyplot(display_image(pred30, f"Prediction {t30}"))
    col2.pyplot(display_image(pred60, f"Prediction {t60}"))

    # =========================
    # GROUND TRUTH UPLOAD
    # =========================

    st.subheader("Upload Ground Truth (Optional)")

    gt_files = st.file_uploader(
        "Upload GT images (t+30, t+60)",
        type=["tif"],
        accept_multiple_files=True,
        key="gt"
    )

    if gt_files and len(gt_files) == 2:

        gt30 = np.array(Image.open(gt_files[0]))
        gt60 = np.array(Image.open(gt_files[1]))

        st.subheader("Comparison")

        # t+30
        col1, col2, col3 = st.columns(3)

        col1.pyplot(display_image(pred30, "Pred t+30"))
        col2.pyplot(display_image(gt30, "GT t+30"))
        col3.pyplot(display_image(np.abs(pred30 - preprocess(gt30)), "Error Map"))

        # t+60
        col1, col2, col3 = st.columns(3)

        col1.pyplot(display_image(pred60, "Pred t+60"))
        col2.pyplot(display_image(gt60, "GT t+60"))
        col3.pyplot(display_image(np.abs(pred60 - preprocess(gt60)), "Error Map"))