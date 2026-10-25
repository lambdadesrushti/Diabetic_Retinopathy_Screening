"""
dr_core.py  -  shared code for the whole project.

Used by the training notebooks (Person A / Person B) AND by the Streamlit app (Person C),
so preprocessing and model definitions can never drift apart.

Contents
  1. constants + preprocessing (CLAHE)
  2. model definitions: classifier, lesion-segmentation U-Net, landmark localizer
  3. Grad-CAM
  4. visualisation helpers
  5. DRPipeline: loads the three trained models and analyses one retina image
"""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ----------------------------------------------------------------------------- 1. constants
GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
LESIONS = ["Microaneurysms", "Haemorrhages", "Hard exudates", "Soft exudates", "Optic disc"]
LESION_SUFFIX = ["MA", "HE", "EX", "SE", "OD"]          # IDRiD mask file suffixes, same order
LESION_COLORS = [(255, 235, 59), (244, 67, 54), (255, 152, 0), (3, 169, 244), (156, 39, 176)]  # RGB

CLS_SIZE, SEG_SIZE, LOC_SIZE = 224, 512, 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLS_FILE, SEG_FILE, LOC_FILE = "dr_classifier.pth", "seg_unet.pth", "loc_resnet.pth"


# ----------------------------------------------------------------------------- 1. preprocessing
def prep_image(img_rgb, size):
    """Squash-resize to size x size, then CLAHE on the L channel. Input/Output: uint8 RGB."""
    img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def to_tensor(img_rgb_uint8):
    """uint8 HxWx3 -> normalised float tensor 1x3xHxW (ImageNet statistics)."""
    x = (img_rgb_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


# ----------------------------------------------------------------------------- 2. models
def build_classifier(pretrained=False):
    """EfficientNet-B0 with a 5-class head (DR grades 0-4)."""
    w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=w)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 5)
    return m


class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class UNetResNet18(nn.Module):
    """Small U-Net with a ResNet-18 encoder. Output: 5 channels (MA, HE, EX, SE, OD) of logits."""

    def __init__(self, n_classes=5, pretrained=False):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.u3, self.u2 = _Up(512, 256, 256), _Up(256, 128, 128)
        self.u1, self.u0 = _Up(128, 64, 64), _Up(64, 64, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        s0 = self.stem(x)
        x1 = self.l1(self.pool(s0))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        d = self.u0(self.u1(self.u2(self.u3(x4, x3), x2), x1), s0)
        return F.interpolate(self.head(d), size=(H, W), mode="bilinear", align_corners=False)


def build_localizer(pretrained=False):
    """ResNet-18 regressor -> (od_x, od_y, fovea_x, fovea_y), each in 0..1 of image width/height."""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Sequential(nn.Linear(512, 4), nn.Sigmoid())
    return m


def save_half(model, path):
    """Save weights in fp16 (halves file size so everything fits comfortably on GitHub)."""
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save(sd, path)


def load_weights(model, path):
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    return model.eval()


def update_metrics(models_dir, key, values):
    """Merge one block of results into models/metrics.json (shown on the app's About tab)."""
    p = os.path.join(models_dir, "metrics.json")
    data = json.load(open(p)) if os.path.exists(p) else {}
    data[key] = values
    json.dump(data, open(p, "w"), indent=2)


# ----------------------------------------------------------------------------- 3. Grad-CAM
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(x)
            probs = F.softmax(logits, 1)[0].detach()
            idx = int(probs.argmax()) if class_idx is None else class_idx
            logits[0, idx].backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1))[0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, probs.cpu().numpy(), idx


# ----------------------------------------------------------------------------- 4. visualisation
def retina_mask(img_rgb):
    """Boolean mask of the circular retina area (dark border removed)."""
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    m = (g > 15).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return m.astype(bool)


def cam_overlay(img_rgb, cam, alpha=0.5):
    h, w = img_rgb.shape[:2]
    c = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    c = np.clip(c, 0, 1) * retina_mask(img_rgb)
    heat = cv2.cvtColor(cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    a = (alpha * np.clip(c * 1.6, 0, 1))[..., None]
    return (img_rgb * (1 - a) + heat * a).astype(np.uint8)


def lesion_overlay(img_rgb, probs, thr=0.5, show=(True, True, True, True, True)):
    """probs: (5,H,W) float maps already resized to the image size. Returns (overlay, burden_percent[4])."""
    out = img_rgb.astype(np.float32).copy()
    ret = retina_mask(img_rgb)
    area = max(int(ret.sum()), 1)
    burden = []
    for i in range(5):
        m = (probs[i] >= thr) & ret
        if i < 4:
            burden.append(100.0 * m.sum() / area)
        if not show[i] or not m.any():
            continue
        col = np.array(LESION_COLORS[i], dtype=np.float32)
        if i == 4:      # optic disc: draw outline only
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tmp = np.zeros(img_rgb.shape[:2], np.uint8)
            cv2.drawContours(tmp, cnts, -1, 1, 2)
            out[tmp.astype(bool)] = col
        else:
            out[m] = out[m] * 0.35 + col * 0.65
    return out.astype(np.uint8), burden


def draw_landmarks(img_rgb, loc):
    """loc = (od_x, od_y, fv_x, fv_y) normalised. Draws labelled markers."""
    out = img_rgb.copy()
    h, w = out.shape[:2]
    r = max(10, int(min(h, w) * 0.035))
    for (nx, ny), name, col in ((loc[0:2], "Optic disc", (156, 39, 176)), (loc[2:4], "Fovea", (0, 200, 120))):
        x, y = int(nx * w), int(ny * h)
        cv2.circle(out, (x, y), r + 3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(out, (x, y), r, col, 3, cv2.LINE_AA)
        cv2.line(out, (x - r - 8, y), (x + r + 8, y), col, 2, cv2.LINE_AA)
        cv2.line(out, (x, y - r - 8), (x, y + r + 8), col, 2, cv2.LINE_AA)
        tx, ty = min(x + r + 12, w - 130), max(y - r - 8, 22)
        cv2.putText(out, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return out


def image_quality(img_rgb):
    """Very simple quality check: sharpness (Laplacian variance) + brightness."""
    g = cv2.cvtColor(cv2.resize(img_rgb, (512, 512)), cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    bright = float(g[g > 15].mean()) if (g > 15).any() else 0.0
    if bright < 40:
        label, ok = "Too dark", False
    elif sharp < 15:
        label, ok = "Blurry", False
    else:
        label, ok = "Good", True
    return {"label": label, "ok": ok, "sharpness": sharp, "brightness": bright}


# ----------------------------------------------------------------------------- 5. pipeline
def find_models_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.environ.get("DR_MODELS_DIR"), os.path.join(here, "..", "models"), os.path.join(here, "models")):
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(here, "..", "models"))


class DRPipeline:
    """Loads whichever of the three models exist. Missing models just switch that feature off."""

    def __init__(self, models_dir=None):
        self.dir = models_dir or find_models_dir()
        self.cls = self.seg = self.loc = None
        self.cam = None
        self._load(CLS_FILE, build_classifier, "cls")
        self._load(SEG_FILE, lambda: UNetResNet18(5), "seg")
        self._load(LOC_FILE, build_localizer, "loc")
        if self.cls is not None:
            self.cam = GradCAM(self.cls, self.cls.features[-1])
        mp = os.path.join(self.dir, "metrics.json")
        self.metrics = json.load(open(mp)) if os.path.exists(mp) else {}

    def _load(self, fname, builder, attr):
        p = os.path.join(self.dir, fname)
        if os.path.exists(p):
            try:
                setattr(self, attr, load_weights(builder(), p))
            except Exception as e:                      # corrupt / mismatched file
                print(f"[dr_core] could not load {fname}: {e}")

    @property
    def status(self):
        return {"classifier": self.cls is not None, "segmentation": self.seg is not None,
                "localization": self.loc is not None}

    def analyze(self, img_rgb, max_side=720):
        """Run every available model once. Returns raw maps; thresholds are applied later (live slider)."""
        h, w = img_rgb.shape[:2]
        s = max_side / max(h, w)
        disp = cv2.resize(img_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        res = {"display": disp, "quality": image_quality(disp)}
        dh, dw = disp.shape[:2]

        if self.cls is not None:
            cam, probs, idx = self.cam(to_tensor(prep_image(img_rgb, CLS_SIZE)))
            res.update(probs=probs.tolist(), grade=idx, confidence=float(probs[idx]), cam=cam)

        with torch.no_grad():
            if self.seg is not None:
                p = torch.sigmoid(self.seg(to_tensor(prep_image(img_rgb, SEG_SIZE))))[0].numpy()
                res["seg"] = np.stack([cv2.resize(c, (dw, dh), interpolation=cv2.INTER_LINEAR) for c in p]).astype(np.float16)
            if self.loc is not None:
                res["loc"] = self.loc(to_tensor(prep_image(img_rgb, LOC_SIZE)))[0].numpy().tolist()
        return res
