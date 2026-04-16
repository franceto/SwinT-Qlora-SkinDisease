# Phân Loại Bệnh Ngoài Da với SwinT + QLoRA

Dự án phân loại 10 bệnh ngoài da sử dụng mô hình Swin Transformer (Swin-T) được fine-tune bằng kỹ thuật QLoRA trên bộ dữ liệu Skin Diseases Image Dataset từ Kaggle.

---

## Mục Lục

- [Bài Toán](#bài-toán)
- [Bộ Dữ Liệu](#bộ-dữ-liệu)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Cài Đặt Môi Trường](#cài-đặt-môi-trường)
- [Cài Đặt Dataset](#cài-đặt-dataset)
- [Siêu Tham Số](#siêu-tham-số)
- [Huấn Luyện Mô Hình](#huấn-luyện-mô-hình)
- [Kết Quả](#kết-quả)
- [Demo Streamlit](#demo-streamlit)
- [Hướng Dẫn Push Lên GitHub](#hướng-dẫn-push-lên-github)

---

## Bài Toán

Phân loại ảnh da liễu vào 10 nhóm bệnh ngoài da phổ biến, bao gồm cả các bệnh lành tính và ác tính. Đây là bài toán phân loại đa lớp (multi-class classification) với dữ liệu mất cân bằng (imbalanced dataset).

**Kiến trúc:**
- Backbone: Swin Transformer Tiny (Swin-T) pretrained trên ImageNet
- Fine-tune: QLoRA (Quantized Low-Rank Adaptation) — chỉ train 0.77% tham số
- Loss: Focal Loss với class weights
- Augmentation: Albumentations (flip, rotate, brightness, contrast, CoarseDropout)
- Xử lý mất cân bằng: WeightedRandomSampler + Focal Loss

---

## Bộ Dữ Liệu

**Nguồn:** [Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)

**Tổng số ảnh:** ~27.200 ảnh | **Số lớp:** 10

| STT | Tên Lớp | Số Ảnh |
|-----|---------|--------|
| 1 | Chàm (Eczema) | ~1.677 |
| 2 | Ung thư hắc tố (Melanoma) | ~15.750 |
| 3 | Viêm da dị ứng (Atopic Dermatitis) | ~1.250 |
| 4 | Ung thư tế bào đáy (BCC) | ~3.323 |
| 5 | Nốt ruồi (Melanocytic Nevi) | ~7.970 |
| 6 | Tổn thương sừng hóa lành tính (BKL) | ~2.624 |
| 7 | Vảy nến và Địa y phẳng (Psoriasis) | ~2.000 |
| 8 | Dày sừng tiết bã và U lành tính | ~1.800 |
| 9 | Nấm da và Hắc lào | ~1.700 |
| 10 | Mụn cóc và Nhiễm virus | ~2.103 |

**Chia dữ liệu:** Train 80% / Val 10% / Test 10% (stratified)

---

## Cấu Trúc Dự Án

```
SkinDisease_SwinT/
├── data/                          # Dataset (không đẩy lên GitHub)
│   └── IMG_CLASSES/
│       ├── 1. Eczema 1677/
│       ├── 2. Melanoma 15.75k/
│       └── ...
├── models/                        # Mô hình đã huấn luyện (không đẩy lên GitHub)
│   └── best_swint_skin.pth
├── notebooks/                     # Jupyter Notebook huấn luyện
│   └── SkinDisease_SwinT.ipynb
├── logs/                          # Biểu đồ và log huấn luyện
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── size_distribution.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── test_predictions.png
├── predictions/                   # Kết quả dự đoán
├── streamlit_app/
│   └── app.py                     # Ứng dụng demo Streamlit
├── requirements/
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

## Cài Đặt Môi Trường

### Yêu Cầu Hệ Thống

- Python 3.10+
- CUDA 12.6+ (GPU NVIDIA, khuyến nghị VRAM >= 4GB)
- Windows 10/11

### Tạo Môi Trường Ảo

```bash
python -m venv venv
venv\Scripts\activate
```

### Cài Đặt PyTorch (CUDA 12.6)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Cài Đặt Thư Viện

```bash
pip install -r requirements/requirements.txt
```

**Nội dung `requirements.txt`:**

```
peft
timm
pandas
numpy
matplotlib
seaborn
scikit-learn
opencv-python
albumentations
kaggle
streamlit
Pillow
tqdm
torchmetrics
```

---

## Cài Đặt Dataset

### Bước 1 — Cấu Hình Kaggle API

1. Truy cập [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**
2. Đặt file `kaggle.json` vào thư mục `requirements/kaggle.json`

### Bước 2 — Tải Dataset

Chạy cell tải dataset trong notebook, hoặc dùng lệnh:

```bash
set KAGGLE_CONFIG_DIR=requirements
kaggle datasets download -d ismailpromus/skin-diseases-image-dataset -p data --unzip
```

Dataset sẽ được giải nén vào `data/IMG_CLASSES/`.

---

## Siêu Tham Số

| Tham Số | Giá Trị | Mô Tả |
|---------|---------|-------|
| `model` | swin_tiny_patch4_window7_224 | Backbone Swin-T pretrained ImageNet |
| `img_size` | 224 | Kích thước ảnh đầu vào |
| `batch_size` | 32 | Kích thước batch |
| `epochs` | 50 | Số epoch tối đa |
| `lr_head` | 3e-4 | Learning rate cho head và LoRA layers |
| `lr_backbone` | 1e-5 | Learning rate cho backbone |
| `weight_decay` | 1e-2 | Hệ số regularization AdamW |
| `patience` | 5 | Số epoch chờ trước Early Stopping |
| `lora_r` | 8 | Rank của LoRA |
| `lora_alpha` | 16 | Scaling factor LoRA |
| `lora_dropout` | 0.1 | Dropout của LoRA |
| `num_classes` | 10 | Số lớp bệnh |

**Tham số có thể train:** 213.120 / 27.740.164 (0.77%)

---

## Huấn Luyện Mô Hình

### Chạy Notebook

Mở file notebook trong VSCode hoặc Jupyter:

```
notebooks/SkinDisease_SwinT.ipynb
```

Chạy lần lượt từng cell theo thứ tự:

| Cell | Nội Dung |
|------|---------|
| Cell 0 | Kiểm tra GPU |
| Cell 1 | Import thư viện |
| Cell 2 | Tải dataset từ Kaggle |
| Cell 3 | Kiểm tra số lượng ảnh từng class |
| Cell 4 | EDA — Biểu đồ phân bố, ảnh đại diện, kích thước ảnh |
| Cell 5 | Tiền xử lý và chia dữ liệu |
| Cell 6 | Cấu hình siêu tham số |
| Cell 7 | Xây dựng mô hình SwinT + QLoRA |
| Cell 8 | Huấn luyện với AMP (mixed precision) |
| Cell 9 | Biểu đồ loss / accuracy / F1 / AUC |
| Cell 10 | Đánh giá trên tập test |
| Cell 11 | Hiển thị 10 ảnh dự đoán ngẫu nhiên |
| Cell 12 | Xuất file app.py Streamlit |

### Vị Trí Kết Quả

| Kết Quả | Đường Dẫn |
|---------|-----------|
| Mô hình tốt nhất | `models/best_swint_skin.pth` |
| Biểu đồ phân bố class | `logs/class_distribution.png` |
| Ảnh đại diện từng class | `logs/sample_images.png` |
| Biểu đồ kích thước ảnh | `logs/size_distribution.png` |
| Biểu đồ quá trình huấn luyện | `logs/training_curves.png` |
| Confusion matrix | `logs/confusion_matrix.png` |
| Ảnh dự đoán mẫu | `logs/test_predictions.png` |

---

## Kết Quả

Kết quả đánh giá trên tập test (~2.716 ảnh):

| Chỉ Số | Giá Trị |
|--------|---------|
| Accuracy | 76.8% |
| Macro F1 | 75.0% |
| Macro AUC | 98.3% |
| Val Loss tốt nhất | 0.2518 |

Early Stopping tại epoch 28 (patience=5).

---

## Demo Streamlit

### Tính Năng

- Tải ảnh da liễu từ máy tính (hỗ trợ `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)
- Hiển thị ảnh gốc đầy đủ ngữ cảnh
- Nút **Dự đoán** chạy inference trên GPU/CPU
- Hiển thị tên bệnh dự đoán và độ tin cậy
- Hiển thị xác suất toàn bộ 10 lớp bệnh
- Biểu đồ cột ngang xác suất từng lớp

### Chạy Ứng Dụng

```bash
cd "D:\Documents\HK2 2025-26 KHDL (Y3) Ky 6\Đồ án CN\SkinDisease_SwinT"
venv\Scripts\activate
streamlit run streamlit_app/app.py
```

Truy cập trình duyệt tại: http://localhost:8501

---

## Hướng Dẫn Push Lên GitHub

### Bước 1 — Tạo file `.gitignore`

Tạo file `.gitignore` tại thư mục gốc dự án với nội dung:

```
venv/
data/
models/
requirements/kaggle.json
__pycache__/
.ipynb_checkpoints/
*.pyc
*.pth
*.h5
```

### Bước 2 — Khởi Tạo Git

```bash
cd "D:\Documents\HK2 2025-26 KHDL (Y3) Ky 6\Đồ án CN\SkinDisease_SwinT"
git init
git add .
git commit -m "Initial commit: SkinDisease SwinT + QLoRA"
```

### Bước 3 — Tạo Repository Trên GitHub

1. Truy cập [github.com](https://github.com) → **New repository**
2. Đặt tên repo, chọn **Public** hoặc **Private**
3. Không tích chọn "Add README" (đã có sẵn)

### Bước 4 — Đẩy Code Lên GitHub

```bash
git remote add origin https://github.com/<username>/<repo-name>.git
git branch -M main
git push -u origin main
```

---

## Công Nghệ Sử Dụng

| Thư Viện | Mục Đích |
|----------|---------|
| PyTorch + CUDA 12.6 | Framework deep learning |
| timm | Swin Transformer pretrained |
| PEFT | QLoRA fine-tuning |
| Albumentations | Data augmentation |
| torchmetrics | Đánh giá F1, AUC |
| Streamlit | Demo web app |
| scikit-learn | Confusion matrix, classification report |
| Pillow / OpenCV | Đọc và xử lý ảnh |
