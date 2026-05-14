# Multimodal Sentiment Analysis on Twitter-2015

Phân tích cảm xúc đa phương thức (văn bản + hình ảnh) trên tập dữ liệu Twitter-2015 ABSA.  
So sánh 4 kiến trúc: **BERTweet**, **ResNet50**, **BERTweet + ResNet50 (Early Fusion)**, **BERTweet + CLIP**.

---

## Giới thiệu

Dự án nghiên cứu bài toán **Aspect-Based Sentiment Analysis (ABSA)** trên mạng xã hội Twitter, nơi mỗi mẫu gồm một cặp (tweet + ảnh đính kèm + aspect target). Mục tiêu phân loại cảm xúc thành 3 nhãn: **Tích cực**, **Trung lập**, **Tiêu cực**.

Điểm nhấn là so sánh hiệu quả của các visual encoder (ResNet50 vs. CLIP) khi kết hợp với BERTweet, đánh giá khi nào hình ảnh thực sự bổ sung thông tin hữu ích cho bài toán sentiment.

---

## Kết quả thực nghiệm

| Mô hình | Accuracy | F1 (Macro) | F1 (Weighted) |
|---|---|---|---|
| BERTweet Text-Only | 76.95% | — | — |
| ResNet50 Image-Only | 50.5% | — | — |
| BERTweet + ResNet50 (Early Fusion) | 76.47% | — | — |
| **BERTweet + CLIP (Fusion)** | **77.05%** | — | — |



**Nhận xét:** BERTweet + CLIP đạt accuracy cao nhất, vượt Early Fusion BERTweet+ResNet50 khoảng 0.58% và BERTweet Text-Only khoảng 0.10%. Chất lượng visual encoder là yếu tố then chốt — CLIP hiểu ngữ nghĩa ảnh tốt hơn ResNet50 thuần trong ngữ cảnh mạng xã hội.

---

## Cấu trúc thư mục

```
.
├── phantichcamxuc.ipynb                         # Notebook huấn luyện 4 mô hình
├── models.py                                    # Định nghĩa kiến trúc mô hình
├── app.py                                       # Streamlit demo web app
├── requirements.txt
├── README.md
└── outputs/                                     # Tự sinh sau khi chạy notebook
    ├── bertweet_text_only.pth
    ├── resnet50_image_only.pth
    ├── early_fusion_stable_bertweet_resnet50.pth
    └── bertweet_clip_fusion.pth
```

---

## Dataset

**Twitter-2015** — tập dữ liệu ABSA tiêu chuẩn cho multimodal sentiment analysis.

| Split | Số mẫu |
|---|---|
| Train | ~7,038 |
| Validation | ~2,303 |
| Test | ~2,149 |

Mỗi mẫu gồm: tweet text, ảnh đính kèm, aspect target, nhãn cảm xúc (0 = Negative / 1 = Neutral / 2 = Positive).

> Dataset không được đính kèm trong repo. Tải về tại Kaggle và đặt theo cấu trúc:
> ```
> datasets/
> ├── twitter2015/
> │   ├── train.tsv
> │   ├── dev.tsv
> │   └── test.tsv
> └── twitter2015_images/
> ```

---

## 4 mô hình

| Mô hình | Đầu vào | Mô tả |
|---|---|---|
| BERTweet Text-Only | Text | Fine-tune `vinai/bertweet-base` + classification head (Linear → GELU → Linear) |
| ResNet50 Image-Only | Ảnh | ResNet50 `IMAGENET1K_V2`, thay `fc` bằng classification head tùy chỉnh |
| BERTweet + ResNet50 | Text + Ảnh | Early fusion: nối `[CLS]` token với image feature sau khi chiếu cùng chiều |
| BERTweet + CLIP | Text + Ảnh | Fusion BERTweet với vision encoder của `openai/clip-vit-base-patch32` |

---

## Kỹ thuật chính

- **Differential learning rate** — BERT/CLIP encoder dùng lr thấp (2e-5), classification head dùng lr cao hơn (1e-4)
- **Class-weighted loss + Label smoothing** — xử lý mất cân bằng nhãn trong dataset
- **Gradient accumulation** — ổn định training với batch nhỏ cho multimodal model
- **Weight transfer** — khởi tạo trọng số Early Fusion từ 2 mô hình đơn đã pre-train
- **Data augmentation** — RandomHorizontalFlip, RandomRotation cho ảnh ở tập train

---

## Cài đặt

```bash
pip install -r requirements.txt
```

**Huấn luyện** — chạy toàn bộ `phantichcamxuc.ipynb` trên Kaggle/Colab (khuyến nghị GPU P100 trở lên).

**Demo web:**

```bash
# Đặt file .pth cùng thư mục với app.py
streamlit run app.py
```

Truy cập `http://localhost:8501`, chọn mô hình, nhập text và upload ảnh để xem kết quả phân tích cảm xúc.

---

## Môi trường

| | |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA Tesla P100 16GB |
| PyTorch | 2.x |
| Transformers | HuggingFace ≥ 4.40 |
| Epochs | 5–6 |
| Batch size | 16 (text/image), 8 (multimodal) |

---


- Zhang et al. (2018). Adaptive Co-attention Network for Named Entity Recognition in Tweets *(Twitter-2015 dataset)*. AAAI 2018.
