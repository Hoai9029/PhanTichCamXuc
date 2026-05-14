"""
models.py - Định nghĩa các kiến trúc mô hình phân tích cảm xúc
Bao gồm: BERTweet, ResNet50, BERTweet+ResNet50 (Early Fusion), BERTweet+CLIP
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from transformers import AutoModel, CLIPModel

BERT_NAME = "vinai/bertweet-base"
CLIP_NAME = "openai/clip-vit-base-patch32"
NUM_CLASSES = 3
DROPOUT = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: BERTweet (Text Only)
# ─────────────────────────────────────────────────────────────────────────────
class TextOnlyBERTweet(nn.Module):
    """Mô hình chỉ dùng văn bản (BERTweet)."""

    def __init__(self, bert_name=BERT_NAME, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch):
        out = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: ResNet50 (Image Only)
# ─────────────────────────────────────────────────────────────────────────────
class ImageOnlyResNet50(nn.Module):
    """Mô hình chỉ dùng hình ảnh (ResNet50)."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        backbone = tvm.resnet50(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, batch):
        feat = self.backbone(batch["image"])
        return self.head(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: BERTweet + ResNet50 (Early Fusion)
# ─────────────────────────────────────────────────────────────────────────────
class EarlyFusionBERTResNet(nn.Module):
    """Early fusion: nối đặc trưng BERTweet và ResNet50 rồi phân loại."""

    def __init__(self, bert_name=BERT_NAME, num_classes=NUM_CLASSES, dropout=0.15):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_hidden = self.bert.config.hidden_size

        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        resnet = tvm.resnet50(weights=weights)
        img_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.resnet = resnet

        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, bert_hidden),
            nn.LayerNorm(bert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(bert_hidden * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch):
        text_out = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        text_cls = text_out.last_hidden_state[:, 0, :]

        img_feat = self.resnet(batch["image"])
        img_feat = self.img_proj(img_feat)

        fused = torch.cat([text_cls, img_feat], dim=1)
        return self.fusion(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Model 4: BERTweet + CLIP Fusion
# ─────────────────────────────────────────────────────────────────────────────
class BERTweetCLIPFusion(nn.Module):
    """Kết hợp BERTweet (text) và CLIP Vision (image)."""

    def __init__(
        self,
        bert_name=BERT_NAME,
        clip_name=CLIP_NAME,
        num_classes=NUM_CLASSES,
        dropout=0.15,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.clip = CLIPModel.from_pretrained(clip_name)

        bert_dim = self.bert.config.hidden_size
        clip_dim = self.clip.config.vision_config.hidden_size  # 768

        # Không fine-tune text encoder của CLIP
        for p in self.clip.text_model.parameters():
            p.requires_grad = False

        self.img_proj = nn.Sequential(
            nn.Linear(clip_dim, bert_dim),
            nn.LayerNorm(bert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(bert_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch):
        text_out = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        text_feat = text_out.last_hidden_state[:, 0, :]

        image_outputs = self.clip.vision_model(pixel_values=batch["pixel_values"])
        image_feat = image_outputs.pooler_output
        image_feat = self.img_proj(image_feat)

        fused = torch.cat([text_feat, image_feat], dim=1)
        return self.fusion(fused)
