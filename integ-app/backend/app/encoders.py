# src/encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 256

def load_image(path_or_pil):
    if isinstance(path_or_pil, Image.Image):
        return path_or_pil.convert("RGB")
    return Image.open(path_or_pil).convert("RGB")


class TextEncoder(nn.Module):
    def __init__(self, projector_path: str | None = None, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device)
        hidden = self.model.config.hidden_size
        self.projector = nn.Linear(hidden, EMBED_DIM).to(device)

        if projector_path:
            state = torch.load(projector_path, map_location=device)
            self.projector.load_state_dict(state)
        self.eval()

    @torch.no_grad()
    def encode(self, texts, normalize: bool = True) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)  # (B, H)
        proj = self.projector(emb)  # (B, 256)
        if normalize:
            proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)
        return proj


class ImageEncoder(nn.Module):
    def __init__(self, projector_path: str | None = None, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        base.eval()
        self.features = base.features.to(device)
        self.projector = nn.Linear(576, EMBED_DIM).to(device)

        if projector_path:
            state = torch.load(projector_path, map_location=device)
            self.projector.load_state_dict(state)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        self.eval()

    @torch.no_grad()
    def encode(self, images, normalize: bool = True) -> torch.Tensor:
        single = False
        if not isinstance(images, (list, tuple)):
            images = [images]
            single = True

        tensors = []
        for img in images:
            pil_img = load_image(img)
            tensors.append(self.transform(pil_img))
        batch = torch.stack(tensors, dim=0).to(self.device)

        feat = self.features(batch)         # (B, 576, 7, 7)
        feat = feat.mean(dim=[2, 3])        # (B, 576)
        proj = self.projector(feat)         # (B, 256)
        if normalize:
            proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)

        return proj if not single else proj[0:1]
