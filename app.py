"""
app.py - Streamlit Demo: Phân tích cảm xúc đa phương thức
Hỗ trợ 4 mô hình: BERTweet, ResNet50, BERTweet+ResNet50, BERTweet+CLIP
"""

import gc
import os

import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor

from models import (
    BERTweetCLIPFusion,
    EarlyFusionBERTResNet,
    ImageOnlyResNet50,
    TextOnlyBERTweet,
)

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
BERT_NAME = "vinai/bertweet-base"
CLIP_NAME = "openai/clip-vit-base-patch32"
MAX_LEN = 128
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES = ["Tiêu cực 😠", "Trung lập 😐", "Tích cực 😊"]
LABEL_COLORS = ["#ef4444", "#f59e0b", "#22c55e"]  # red, amber, green

MODEL_INFO = {
    "BERTweet (Text Only)": {
        "key": "bertweet",
        "needs_image": False,
        "weight_file": "bertweet_text_only.pth",
        "description": "Mô hình chỉ dùng văn bản, dựa trên BERTweet pre-trained trên Twitter.",
        "icon": "📝",
    },
    "ResNet50 (Image Only)": {
        "key": "resnet",
        "needs_image": True,
        "weight_file": "resnet50_image_only.pth",
        "description": "Mô hình chỉ dùng hình ảnh, dựa trên ResNet50 pre-trained trên ImageNet.",
        "icon": "🖼️",
    },
    "BERTweet + ResNet50 (Fusion)": {
        "key": "bertweet_resnet",
        "needs_image": True,
        "weight_file": "early_fusion_stable_bertweet_resnet50.pth",
        "description": "Early fusion kết hợp đặc trưng văn bản (BERTweet) và hình ảnh (ResNet50).",
        "icon": "🔀",
    },
    "BERTweet + CLIP (Fusion)": {
        "key": "bertweet_clip",
        "needs_image": True,
        "weight_file": "bertweet_clip_fusion.pth",
        "description": "Fusion BERTweet và CLIP Vision encoder cho hiểu biết đa phương thức.",
        "icon": "✨",
    },
}

IMG_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Cache tokenizers & processors (chỉ load 1 lần)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải tokenizer BERTweet...")
def load_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_NAME, normalization=True, use_fast=False)


@st.cache_resource(show_spinner="Đang tải CLIP processor...")
def load_clip_processor():
    return CLIPProcessor.from_pretrained(CLIP_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Load model theo lựa chọn (cache riêng từng model)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_model(model_key: str, weight_file: str):
    """Load model và weights, trả về model ở eval mode."""
    if model_key == "bertweet":
        model = TextOnlyBERTweet()
    elif model_key == "resnet":
        model = ImageOnlyResNet50()
    elif model_key == "bertweet_resnet":
        model = EarlyFusionBERTResNet()
    elif model_key == "bertweet_clip":
        model = BERTweetCLIPFusion()
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    if os.path.exists(weight_file):
        state_dict = torch.load(weight_file, map_location=DEVICE)
        model.load_state_dict(state_dict)
        st.toast(f"✅ Đã tải weights: {weight_file}", icon="✅")
    else:
        st.warning(
            f"⚠️ Không tìm thấy file weights `{weight_file}`. "
            "Mô hình chạy với trọng số mặc định (chưa train).",
            icon="⚠️",
        )

    model.to(DEVICE)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tiền xử lý đầu vào
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str, tokenizer):
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"].to(DEVICE),
        "attention_mask": enc["attention_mask"].to(DEVICE),
    }


def preprocess_image_resnet(pil_image: Image.Image):
    tensor = IMG_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    return {"image": tensor}


def preprocess_image_clip(pil_image: Image.Image, clip_processor):
    inputs = clip_processor(images=pil_image.convert("RGB"), return_tensors="pt")
    return {"pixel_values": inputs["pixel_values"].to(DEVICE)}


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, batch: dict):
    logits = model(batch)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


# ─────────────────────────────────────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────────────────────────────────────
def render_result(pred_idx: int, probs: np.ndarray):
    label = LABEL_NAMES[pred_idx]
    color = LABEL_COLORS[pred_idx]
    confidence = probs[pred_idx] * 100

    st.markdown("---")
    st.subheader("📊 Kết quả phân tích")

    # Kết quả chính
    st.markdown(
        f"""
        <div style="
            background: {color}22;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <div style="font-size: 2.5rem; font-weight: 700; color: {color};">
                {label}
            </div>
            <div style="font-size: 1.1rem; color: #6b7280; margin-top: 4px;">
                Độ tin cậy: <b>{confidence:.1f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Thanh xác suất từng nhãn
    st.markdown("**Phân phối xác suất:**")
    raw_labels = ["Tiêu cực", "Trung lập", "Tích cực"]
    for i, (raw_label, prob, color_hex) in enumerate(
        zip(raw_labels, probs, LABEL_COLORS)
    ):
        pct = prob * 100
        bar_html = f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span style="font-size: 0.9rem; font-weight: {'700' if i == pred_idx else '400'};">
                    {LABEL_NAMES[i]}
                </span>
                <span style="font-size: 0.9rem; color: #6b7280;">{pct:.1f}%</span>
            </div>
            <div style="background: #e5e7eb; border-radius: 6px; height: 10px;">
                <div style="
                    background: {color_hex};
                    width: {pct:.1f}%;
                    height: 10px;
                    border-radius: 6px;
                    transition: width 0.5s ease;
                "></div>
            </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Phân Tích Cảm Xúc Đa Phương Thức",
        page_icon="🎭",
        layout="wide",
    )

    # Header
    st.markdown(
        """
        <h1 style="text-align: center; color: #1e293b;">
             Phân Tích Cảm Xúc Đa Phương Thức
        </h1>
        <p style="text-align: center; color: #6b7280; font-size: 1.05rem;">
            So sánh 4 mô hình: BERTweet · ResNet50 · BERTweet+ResNet50 · BERTweet+CLIP
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Sidebar: chọn mô hình ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Cài đặt")

        selected_model_name = st.selectbox(
            "🤖 Chọn mô hình",
            options=list(MODEL_INFO.keys()),
            index=0,
        )

        info = MODEL_INFO[selected_model_name]
        st.info(f"{info['icon']} {info['description']}")

        needs_image = info["needs_image"]
        if needs_image:
            st.success("📷 Mô hình này yêu cầu **cả text lẫn ảnh**.")
        else:
            st.warning("📝 Mô hình này chỉ dùng **văn bản**, không cần ảnh.")

        st.markdown("---")

        # Nút làm mới để xóa cache model cũ khi đổi mô hình
        if st.button("🔄 Làm mới / Đổi mô hình", use_container_width=True, type="secondary"):
            # Xóa cache của load_model để giải phóng RAM/VRAM
            load_model.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st.session_state.pop("last_result", None)
            st.success("✅ Đã giải phóng bộ nhớ. Mô hình mới sẽ được tải khi bạn nhấn Phân tích.")
            st.rerun()

        st.markdown("---")
        st.caption(f"🖥️ Thiết bị: `{DEVICE}`")
        st.caption("Twitter-2015 · Phân tích ABSA")

    # ── Phần nhập liệu ────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("✏️ Nhập văn bản")
        text_input = st.text_area(
            "Nhập tweet / câu văn cần phân tích:",
            placeholder="Ví dụ: I love the new iPhone! The camera is amazing.",
            height=140,
            label_visibility="collapsed",
        )

        if needs_image:
            st.subheader("🖼️ Tải lên hình ảnh")
            uploaded_file = st.file_uploader(
                "Chọn ảnh (JPG/PNG):",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
            if uploaded_file:
                pil_image = Image.open(uploaded_file).convert("RGB")
                st.image(pil_image, caption="Ảnh đầu vào", use_container_width=True)
            else:
                pil_image = None
        else:
            pil_image = None

    with col2:
        st.subheader("🚀 Chạy phân tích")

        run_btn = st.button(
            f"{info['icon']} Phân tích với {selected_model_name}",
            use_container_width=True,
            type="primary",
        )

        # Validate
        can_run = True
        if not text_input.strip():
            st.warning("⚠️ Vui lòng nhập văn bản.", icon="⚠️")
            can_run = False
        if needs_image and pil_image is None:
            st.warning("⚠️ Mô hình này cần hình ảnh. Vui lòng tải ảnh lên.", icon="⚠️")
            can_run = False

        if run_btn and can_run:
            with st.spinner(f"Đang tải mô hình và phân tích..."):
                try:
                    # Load tokenizer / processor
                    tokenizer = load_tokenizer()

                    # Load model (có cache)
                    model = load_model(info["key"], info["weight_file"])

                    # Tạo batch
                    batch = preprocess_text(text_input.strip(), tokenizer)

                    if info["key"] == "resnet":
                        batch.update(preprocess_image_resnet(pil_image))
                    elif info["key"] == "bertweet_resnet":
                        batch.update(preprocess_image_resnet(pil_image))
                    elif info["key"] == "bertweet_clip":
                        clip_proc = load_clip_processor()
                        batch.update(preprocess_image_clip(pil_image, clip_proc))

                    # Chỉ giữ key mà model cần
                    model_keys_map = {
                        "bertweet": ["input_ids", "attention_mask"],
                        "resnet": ["image"],
                        "bertweet_resnet": ["input_ids", "attention_mask", "image"],
                        "bertweet_clip": ["input_ids", "attention_mask", "pixel_values"],
                    }
                    needed_keys = model_keys_map[info["key"]]
                    batch = {k: v for k, v in batch.items() if k in needed_keys}

                    pred_idx, probs = predict(model, batch)

                    st.session_state["last_result"] = (pred_idx, probs)

                except Exception as e:
                    st.error(f"❌ Lỗi khi chạy mô hình: {e}")
                    st.exception(e)

        # Hiển thị kết quả
        if "last_result" in st.session_state:
            pred_idx, probs = st.session_state["last_result"]
            render_result(pred_idx, probs)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <p style="text-align: center; color: #9ca3af; font-size: 0.85rem;">
            Dataset: Twitter-2015 ABSA &nbsp;|&nbsp;
            BERTweet: <code>vinai/bertweet-base</code> &nbsp;|&nbsp;
            CLIP: <code>openai/clip-vit-base-patch32</code>
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
