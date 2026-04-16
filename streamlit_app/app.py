
import streamlit as st
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import autocast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "1. Eczema 1677",
    "10. Warts Molluscum and other Viral Infections - 2103",
    "2. Melanoma 15.75k",
    "3. Atopic Dermatitis - 1.25k",
    "4. Basal Cell Carcinoma (BCC) 3323",
    "5. Melanocytic Nevi (NV) - 7970",
    "6. Benign Keratosis-like Lesions (BKL) 2624",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k",
]

VI_NAMES = {
    "1. Eczema 1677": "Chàm (Eczema)",
    "10. Warts Molluscum and other Viral Infections - 2103": "Mụn cóc và Nhiễm virus",
    "2. Melanoma 15.75k": "Ung thư hắc tố (Melanoma)",
    "3. Atopic Dermatitis - 1.25k": "Viêm da dị ứng",
    "4. Basal Cell Carcinoma (BCC) 3323": "Ung thư tế bào đáy (BCC)",
    "5. Melanocytic Nevi (NV) - 7970": "Nốt ruồi (Melanocytic Nevi)",
    "6. Benign Keratosis-like Lesions (BKL) 2624": "Tổn thương sừng hóa lành tính",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "Vảy nến và Địa y phẳng",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": "Dày sừng tiết bã và U lành tính",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "Nấm da và Hắc lào",
}

@st.cache_resource
def load_model():
    base = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=10)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["qkv","proj"], bias="none")
    m = get_peft_model(base, lora_cfg).to(DEVICE)
    m.load_state_dict(torch.load(
        r"D:\Documents\HK2 2025-26 KHDL (Y3) Ky 6\Đồ án CN\SkinDisease_SwinT\models\best_swint_skin.pth",
        map_location=DEVICE
    ))
    m.eval()
    return m

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

st.set_page_config(page_title="Phân loại bệnh ngoài da", layout="wide")
st.title("Phân loại bệnh ngoài da")
st.markdown("Sử dụng SwinT + QLoRA để dự đoán bệnh ngoài da")
st.markdown("---")

model = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Tải ảnh lên")
    uploaded = st.file_uploader("Chọn ảnh từ máy tính", type=["jpg","jpeg","png","bmp","webp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Ảnh đã tải lên", width=350)

with col2:
    st.subheader("Kết quả dự đoán")
    if uploaded:
        if st.session_state.get("probs") is not None:
            probs = st.session_state["probs"]
            pred_idx = probs.argmax()
            st.markdown(f"**Bệnh dự đoán:** {VI_NAMES[CLASS_NAMES[pred_idx]]}")
            st.markdown(f"**Độ tin cậy:** {probs[pred_idx]*100:.2f}%")
            st.markdown("**Xác suất từng lớp bệnh:**")
            for i, cn in enumerate(CLASS_NAMES):
                st.write(f"{VI_NAMES[cn]}: **{probs[i]*100:.2f}%**")

st.markdown("---")
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("Dự đoán", use_container_width=True)

if predict_btn and uploaded:
    img_np = np.array(img)
    tensor = transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with autocast():
            out = model(tensor)
    probs = torch.softmax(out.float(), dim=1).squeeze().cpu().numpy()
    st.session_state["probs"] = probs
    st.rerun()

if uploaded and st.session_state.get("probs") is not None:
    probs = st.session_state["probs"]
    vi_labels = [VI_NAMES[cn] for cn in CLASS_NAMES]
    st.markdown("---")
    st.subheader("Biểu đồ xác suất từng lớp bệnh")
    col_chart1, col_chart2, col_chart3 = st.columns([1, 3, 1])
    with col_chart2:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2196F3" if i != probs.argmax() else "#F44336" for i in range(len(probs))]
        ax.barh(vi_labels, probs * 100, color=colors)
        ax.set_xlabel("Xác suất (%)")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        for i, v in enumerate(probs * 100):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
