import os, io, json, glob, hashlib, datetime
import numpy as np
import cv2
from PIL import Image
import streamlit as st

import premium_theme
premium_theme.apply()

st.set_page_config(page_title="DR + Metabolic Risk Suite", page_icon="🩺", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(ROOT, "outputs")
SAMPLES = os.path.join(HERE, "samples")

GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
LEVELS = ["Low", "Elevated", "High", "Urgent"]
LEVEL_COL = ["#22c55e", "#eab308", "#f97316", "#ef4444"]
LESIONS = ["Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates"]
MASK_COL = [(255, 214, 10), (239, 68, 68), (250, 250, 250), (168, 85, 247), (20, 184, 166)]
ADVICE = [
    "No signs of diabetic retinopathy detected. Continue yearly eye screening and good glycaemic control.",
    "Mild changes. Re-screen in 6-12 months and keep HbA1c and blood pressure on target.",
    "Moderate changes. Refer to an ophthalmologist within a few months and review diabetes management.",
    "Severe changes. Prompt ophthalmology referral is recommended.",
    "Proliferative disease suspected. Urgent specialist referral is recommended.",
]
COMBINED_ADVICE = [
    "Low combined risk. Routine yearly screening and lifestyle advice.",
    "Elevated combined risk. Shorter follow-up (6 months) and tighter HbA1c / BP control.",
    "High combined risk. Ophthalmology referral and a medical review of diabetes therapy.",
    "Urgent combined risk. Same-week specialist referral is recommended.",
]

CSS = """
<style>
:root{--acc:#0F6CBD;--acc2:#12A4A4}
.block-container{padding-top:1.4rem;max-width:1280px}
.hero{padding:26px 32px;border-radius:18px;margin-bottom:16px;
 background:linear-gradient(115deg,var(--acc) 0%,var(--acc2) 100%);
 box-shadow:0 8px 24px rgba(15,108,189,.22)}
.hero h1,.hero p{color:#fff !important}
.hero h1{margin:0;font-size:1.85rem;line-height:1.25}
.hero p{margin:.4rem 0 0 0;opacity:.92}
.badge{display:inline-block;padding:3px 12px;border-radius:999px;font-size:.78rem;color:#fff;
 border:1px solid rgba(255,255,255,.45);margin:10px 6px 0 0;background:rgba(255,255,255,.16)}
.card{padding:18px 20px;border-radius:14px;background:var(--secondary-background-color,#fff);
 border:1px solid rgba(128,128,128,.22);box-shadow:0 2px 10px rgba(16,24,40,.07);margin-bottom:14px}
.grade{font-size:1.9rem;font-weight:700;margin:0;color:var(--acc)}
.bar-wrap{background:rgba(128,128,128,.22);border-radius:10px;height:11px;overflow:hidden}
.bar{height:11px;border-radius:10px;background:linear-gradient(90deg,var(--acc),var(--acc2))}
.row{display:flex;justify-content:space-between;font-size:.9rem;margin:10px 0 4px 0}
.note{font-size:.82rem;opacity:.7;margin:0 0 4px 0}
.pill{display:inline-block;padding:4px 14px;border-radius:999px;font-weight:600;color:#fff}
div[data-testid="stMetric"]{background:var(--secondary-background-color,#fff);border:1px solid rgba(128,128,128,.22);
 border-radius:12px;padding:12px 16px;box-shadow:0 2px 8px rgba(16,24,40,.05)}
button[kind="primary"],button[kind="secondaryFormSubmit"]{border-radius:10px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ACCENTS = {
    "Medical blue": ("#0F6CBD", "#12A4A4"),
    "Teal": ("#0F766E", "#0EA5E9"),
    "Indigo": ("#4338CA", "#0EA5E9"),
    "Emerald": ("#047857", "#14B8A6"),
}


# =====================  MODEL LOADING  =====================
import re, sys, glob, importlib, inspect
import torch
import torch.nn as nn
import torchvision.models as tvm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KNOWN_TOP = {"encoder", "decoder", "segmentation_head", "classification_head", "features", "classifier",
             "conv1", "bn1", "fc", "layer1", "layer2", "layer3", "layer4", "backbone", "head"}


def _load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _clean(obj):
    """Plain state_dict with wrapper prefixes (module./model./net./unet.) removed."""
    sd = obj
    if isinstance(sd, dict):
        for k in ("model_state_dict", "state_dict", "model", "net"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    sd = dict(sd)
    for _ in range(3):
        tops = {k.split(".")[0] for k in sd}
        if len(tops) == 1 and list(tops)[0] not in KNOWN_TOP and all("." in k for k in sd):
            sd = {k.split(".", 1)[1]: v for k, v in sd.items()}
        else:
            break
    return sd


def _head_out(sd, default):
    for k in reversed(list(sd.keys())):
        if k.endswith("weight") and getattr(sd[k], "ndim", 0) == 2:
            return int(sd[k].shape[0])
    return default


def _seq_head(sd, prefix, fin, n):
    """Build an nn.Linear / nn.Sequential head that matches the Linear layers found under `prefix`."""
    if (prefix + ".weight") in sd and sd[prefix + ".weight"].ndim == 2:
        o, i = sd[prefix + ".weight"].shape
        return nn.Linear(int(i), int(o))
    idx = sorted({int(k.split(".")[1]) for k in sd
                  if k.startswith(prefix + ".") and k.split(".")[1].isdigit()
                  and k.endswith(".weight") and sd[k].ndim == 2})
    if not idx:
        return nn.Linear(fin, n)
    layers = []
    for i in range(max(idx) + 1):
        if i in idx:
            o, inn = sd["%s.%d.weight" % (prefix, i)].shape
            layers.append(nn.Linear(int(inn), int(o)))
        else:
            layers.append(nn.ReLU() if i < max(idx) else nn.Identity())
    return nn.Sequential(*layers)


def _try(cands, sd):
    """cands = [(name, factory)]. Returns (model or None, message). Picks the exact match or reports the closest."""
    best = None
    for name, fac in cands:
        try:
            m = fac()
        except Exception:
            continue
        if not isinstance(m, nn.Module):
            continue
        try:
            m.load_state_dict(sd, strict=True)
            return m.to(DEVICE).eval(), "loaded (%s)" % name
        except Exception:
            pass
        try:
            r = m.load_state_dict(sd, strict=False)
            score = len(r.missing_keys) + len(r.unexpected_keys)
            if best is None or score < best[0]:
                best = (score, name, len(r.missing_keys), len(r.unexpected_keys))
        except Exception:
            pass
    ks = list(sd.keys())
    msg = "no exact architecture match. checkpoint keys: %s ... (%d keys)" % (", ".join(ks[:3]), len(ks))
    if best:
        msg += " | closest: %s (missing %d, unexpected %d)" % (best[1], best[2], best[3])
    return None, msg


def _flatten_modules(obj, depth=0):
    """Collect nn.Module objects from a return value (module / tuple / list / dict / object attrs)."""
    out = []
    if isinstance(obj, nn.Module):
        out.append(obj)
    elif depth < 2 and isinstance(obj, (list, tuple)):
        for o in obj:
            out += _flatten_modules(o, depth + 1)
    elif depth < 2 and isinstance(obj, dict):
        for o in obj.values():
            out += _flatten_modules(o, depth + 1)
    return out


def _discover(sd):
    """Search the project's own .py files (app/*.py) for an nn.Module whose state_dict keys equal the checkpoint keys."""
    os.environ.setdefault("DR_MODELS_DIR", MODELS)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    ckpt_keys = set(sd.keys())
    nout = None
    for k in reversed(list(sd.keys())):
        if k.endswith("weight") and getattr(sd[k], "ndim", 0) == 4:
            nout = int(sd[k].shape[0])
            break
    kws = [{}]
    if nout:
        kws += [{"n_classes": nout}, {"num_classes": nout}, {"classes": nout}, {"out_channels": nout}, {"n_out": nout}]
    tried = []
    for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        name = os.path.splitext(os.path.basename(f))[0]
        if name.startswith("app"):
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        tried.append(name)
        produced = []
        for cn, cls in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls, nn.Module) and cls.__module__ == mod.__name__:
                for kw in kws:
                    try:
                        produced.append(("%s.%s" % (name, cn), cls(**kw)))
                        break
                    except Exception:
                        continue
        for fn, fun in inspect.getmembers(mod, inspect.isfunction):
            if fun.__module__ != mod.__name__ or fn.startswith("_"):
                continue
            if not re.search(r"load|build|make|create|get|unet|seg", fn.lower()):
                continue
            try:
                req = [p for p in inspect.signature(fun).parameters.values()
                       if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
            except Exception:
                continue
            if req:
                continue
            try:
                for m in _flatten_modules(fun()):
                    produced.append(("%s.%s()" % (name, fn), m))
            except Exception:
                continue
        for label, m in produced:
            try:
                if set(m.state_dict().keys()) == ckpt_keys:
                    m.load_state_dict(sd, strict=True)
                    return m.to(DEVICE).eval(), "loaded (%s)" % label
            except Exception:
                continue
    return None, "searched app/*.py: " + (", ".join(tried) if tried else "nothing importable")


@st.cache_resource(show_spinner=False)
def load_classifier():
    path = os.path.join(MODELS, "dr_classifier.pth")
    if not os.path.exists(path):
        return None, "dr_classifier.pth not found in models/"
    try:
        obj = _load_ckpt(path)
        if isinstance(obj, nn.Module):
            return obj.to(DEVICE).eval(), "loaded (full model)"
        sd = _clean(obj)
        n = _head_out(sd, 5)

        def eff():
            m = tvm.efficientnet_b0(weights=None)
            m.classifier = nn.Sequential(nn.Dropout(0.3), _seq_head(sd, "classifier.1", 1280, n))
            return m

        def eff2():
            m = tvm.efficientnet_b0(weights=None)
            m.classifier = _seq_head(sd, "classifier", 1280, n)
            return m

        cands = [("efficientnet_b0", eff), ("efficientnet_b0 flat head", eff2)]
        try:
            import timm
            cands.append(("timm efficientnet_b0", lambda: timm.create_model("efficientnet_b0", pretrained=False, num_classes=n)))
        except Exception:
            pass
        return _try(cands, sd)
    except Exception as e:
        return None, "load error: " + str(e)[:250]


@st.cache_resource(show_spinner=False)
def load_seg():
    path = os.path.join(MODELS, "seg_unet.pth")
    if not os.path.exists(path):
        return None, "seg_unet.pth not found in models/"
    try:
        obj = _load_ckpt(path)
        if isinstance(obj, nn.Module):
            return obj.to(DEVICE).eval(), "loaded (full model)"
        sd = _clean(obj)
        found, fmsg = _discover(sd)
        if found is not None:
            return found, fmsg
        try:
            import segmentation_models_pytorch as smp
        except Exception:
            return None, "custom U-Net class not found (" + fmsg + ")"
        heads = [k for k in sd if "segmentation_head" in k and k.endswith("weight")]
        ncls = int(sd[heads[0]].shape[0]) if heads else 4
        cands = []
        for arch in ("Unet", "UnetPlusPlus"):
            for enc in ("resnet34", "resnet18", "resnet50", "resnet101", "efficientnet-b0",
                        "efficientnet-b3", "resnext50_32x4d", "mobilenet_v2", "densenet121"):
                cands.append(("%s/%s" % (arch, enc),
                              (lambda a=arch, e=enc: getattr(smp, a)(e, encoder_weights=None, classes=ncls))))
        m, msg = _try(cands, sd)
        if m is None:
            msg += " | " + fmsg
        return m, msg
    except Exception as e:
        return None, "load error: " + str(e)[:250]


@st.cache_resource(show_spinner=False)
def load_loc():
    path = os.path.join(MODELS, "loc_resnet.pth")
    if not os.path.exists(path):
        return None, "loc_resnet.pth not found in models/"
    try:
        obj = _load_ckpt(path)
        if isinstance(obj, nn.Module):
            return obj.to(DEVICE).eval(), "loaded (full model)"
        sd = _clean(obj)
        n = _head_out(sd, 4)
        cands = []
        for nm, fn in (("resnet18", tvm.resnet18), ("resnet34", tvm.resnet34), ("resnet50", tvm.resnet50)):
            def fac(fn=fn):
                m = fn(weights=None)
                m.fc = _seq_head(sd, "fc", m.fc.in_features, n)
                return m
            cands.append((nm, fac))
        return _try(cands, sd)
    except Exception as e:
        return None, "load error: " + str(e)[:250]


# =====================  INFERENCE HELPERS  =====================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_long(arr, side=640):
    h, w = arr.shape[:2]
    s = side / max(h, w)
    if s >= 1:
        return arr
    return cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def preprocess(rgb, size):
    x = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)


def classify_with_gradcam(model, rgb, size):
    """Returns (probs[5], cam[0..1] at size x size)."""
    x = preprocess(rgb, size)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    target = convs[-1]
    feats, grads = [], []

    def fwd(_m, _i, o):
        feats.append(o)
        o.register_hook(lambda g: grads.append(g))

    h = target.register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            model.zero_grad()
            out = model(x)
            probs = torch.softmax(out, 1)[0]
            cls = int(torch.argmax(probs))
            out[0, cls].backward()
    finally:
        h.remove()
    f, g = feats[0].detach(), grads[0].detach()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * f).sum(1))[0].cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (size, size))
    return probs.detach().cpu().numpy(), cam


def cam_overlay(rgb, cam, alpha, cmap):
    h, w = rgb.shape[:2]
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (w, h))), cmap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def run_seg(model, rgb, size, thr):
    h, w = rgb.shape[:2]
    with torch.no_grad():
        o = model(preprocess(rgb, size))
    if isinstance(o, (list, tuple)):
        o = o[0]
    o = o[0].float().cpu()
    c = o.shape[0]
    p = torch.softmax(o, 0)[1:] if c == 5 else torch.sigmoid(o)
    n = p.shape[0]
    names = LESIONS[:n] if n <= 4 else ["Class %d" % (i + 1) for i in range(n)]
    masks = {}
    for nm, ch in zip(names, p.numpy()):
        masks[nm] = cv2.resize(ch, (w, h)) > thr
    return masks


def run_loc(model, rgb, size):
    h, w = rgb.shape[:2]
    with torch.no_grad():
        o = model(preprocess(rgb, size))
    o = o[0].float().cpu().numpy().ravel()[:4]
    if len(o) < 4:
        raise ValueError("loc model must output 4 numbers")
    scale = np.array([w, h, w, h], dtype=np.float32)
    o = o * scale if np.abs(o).max() <= 1.5 else o / size * scale
    return (float(o[0]), float(o[1])), (float(o[2]), float(o[3]))


# ---- classical fallbacks (used only if a model fails to load; clearly labelled) ----
def fallback_lesions(rgb):
    g = cv2.createCLAHE(2.0, (8, 8)).apply(rgb[:, :, 1])
    fm = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 20).astype(np.uint8)
    fm = cv2.erode(fm, np.ones((25, 25), np.uint8))
    dark = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bright = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))

    def thr(a):
        v = a[fm > 0]
        t = np.percentile(v, 97.5) if v.size else 255
        return (a >= max(t, 25)) & (fm > 0)

    return {"Red lesions (preview)": thr(dark), "Bright lesions (preview)": thr(bright)}


def fallback_markers(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    fm = gray > 20
    blur = cv2.GaussianBlur(gray, (0, 0), w / 25).astype(np.float32)
    b = np.where(fm, blur, 0)
    y, x = np.unravel_index(np.argmax(b), b.shape)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - x, yy - y)
    ok = fm & (d > 0.15 * w) & (d < 0.5 * w)
    c = np.where(ok, blur, np.inf)
    fy, fx = np.unravel_index(np.argmin(c), c.shape)
    return (float(x), float(y)), (float(fx), float(fy))


def compose(rgb, masks, show, od, fov, markers_on, mask_alpha=0.5):
    out = rgb.copy()
    for i, (nm, m) in enumerate(masks.items()):
        if show.get(nm, True) and m.any():
            layer = out.copy()
            layer[m] = MASK_COL[i % len(MASK_COL)]
            out = cv2.addWeighted(layer, mask_alpha, out, 1 - mask_alpha, 0)
    if markers_on:
        r = max(10, rgb.shape[1] // 22)
        if od is not None:
            p = (int(round(od[0])), int(round(od[1])))
            cv2.circle(out, p, r, (34, 197, 94), 3)
            cv2.putText(out, "Optic disc", (p[0] + r + 4, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 197, 94), 2)
        if fov is not None:
            p = (int(round(fov[0])), int(round(fov[1])))
            cv2.circle(out, p, r, (59, 130, 246), 3)
            cv2.putText(out, "Fovea", (p[0] + r + 4, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (59, 130, 246), 2)
    return out


def analyze(rgb, cls_size, seg_size, thr):
    """Runs all 3 models. Every stage is isolated so one failure never kills the app."""
    res = {"probs": None, "cam": None, "masks": {}, "od": None, "fov": None,
           "seg_src": "", "loc_src": "", "errors": []}
    cls, _ = load_classifier()
    if cls is not None:
        try:
            res["probs"], res["cam"] = classify_with_gradcam(cls, rgb, cls_size)
        except Exception as e:
            res["errors"].append("Classifier/Grad-CAM: " + str(e)[:200])
    seg, _ = load_seg()
    try:
        if seg is None:
            raise RuntimeError("seg model not loaded")
        res["masks"] = run_seg(seg, rgb, seg_size, thr)
        res["seg_src"] = "Trained U-Net"
    except Exception as e:
        res["masks"] = fallback_lesions(rgb)
        res["seg_src"] = "Classical-CV preview (U-Net unavailable)"
        res["errors"].append("Segmentation: " + str(e)[:200])
    loc, _ = load_loc()
    try:
        if loc is None:
            raise RuntimeError("loc model not loaded")
        res["od"], res["fov"] = run_loc(loc, rgb, cls_size)
        res["loc_src"] = "Trained ResNet"
    except Exception as e:
        res["od"], res["fov"] = fallback_markers(rgb)
        res["loc_src"] = "Classical-CV preview (ResNet unavailable)"
        res["errors"].append("Localization: " + str(e)[:200])
    return res


def enhance(rgb, mode):
    """Display-only image enhancement (never fed to the models)."""
    if mode == "Contrast boost (CLAHE)":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(2.5, (8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    if mode == "Green channel (lesion contrast)":
        g = cv2.createCLAHE(2.5, (8, 8)).apply(rgb[:, :, 1])
        return np.dstack([g, g, g])
    return rgb


# =====================  RISK + FUSION + PDF  =====================
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage


def metabolic(hba1c, sbp, years):
    s = 0
    s += 0 if hba1c < 6.5 else 1 if hba1c < 7.5 else 2 if hba1c < 9.0 else 3
    s += 0 if sbp < 120 else 1 if sbp < 130 else 2 if sbp < 140 else 3
    s += 0 if years < 5 else 1 if years < 10 else 2 if years < 15 else 3
    idx = 0 if s <= 2 else 1 if s <= 4 else 2 if s <= 6 else 3
    return {"score": s, "idx": idx, "level": LEVELS[idx]}


def fuse(grade, met_idx):
    """Decision-level fusion: retina grade and metabolic level are combined AFTER separate scoring."""
    g = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}[int(grade)]
    idx = int(round(0.65 * g + 0.35 * met_idx))
    if grade >= 3:
        idx = max(idx, 2)
    if grade == 4:
        idx = 3
    return min(idx, 3)


def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    buf.seek(0)
    return buf


def build_pdf(patient, grade, probs, risk, fused_idx, orig, cam_img, seg_src):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    ss = getSampleStyleSheet()
    el = [Paragraph("Diabetic Retinopathy + Metabolic Risk Report", ss["Title"]),
          Paragraph("Research prototype - not a medical device. Not for clinical use.", ss["Italic"]),
          Spacer(1, 10)]

    def table(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                               ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        return t

    el.append(Paragraph("Patient details", ss["Heading2"]))
    el.append(table([["Name / ID", patient["name"]], ["Age", str(patient["age"])],
                     ["Date", datetime.date.today().isoformat()]], [5 * cm, 11 * cm]))
    el.append(Paragraph("Retina analysis", ss["Heading2"]))
    rows = [["Predicted grade", "%d - %s" % (grade, GRADES[grade])]]
    for i, p in enumerate(probs):
        rows.append(["P(%s)" % GRADES[i], "%.1f %%" % (100 * p)])
    el.append(table(rows, [5 * cm, 11 * cm]))
    el.append(Spacer(1, 8))
    imgs = [RLImage(to_png_bytes(orig), width=7.5 * cm, height=7.5 * cm * orig.shape[0] / orig.shape[1]),
            RLImage(to_png_bytes(cam_img), width=7.5 * cm, height=7.5 * cm * cam_img.shape[0] / cam_img.shape[1])]
    el.append(Table([imgs, ["Fundus image", "Grad-CAM heatmap"]]))
    el.append(Paragraph("Lesion / landmark source: " + seg_src, ss["Normal"]))
    el.append(Paragraph("Metabolic risk", ss["Heading2"]))
    el.append(table([["HbA1c (%)", "%.1f" % risk["hba"]], ["Systolic BP (mmHg)", str(risk["sbp"])],
                     ["Diabetes duration (years)", str(risk["dur"])],
                     ["Metabolic level", risk["level"]]], [5 * cm, 11 * cm]))
    el.append(Paragraph("Combined recommendation", ss["Heading2"]))
    el.append(Paragraph("<b>%s risk.</b> %s" % (LEVELS[fused_idx], COMBINED_ADVICE[fused_idx]), ss["Normal"]))
    el.append(Spacer(1, 6))
    el.append(Paragraph("Retina and metabolic scores were computed separately and combined at decision level; "
                        "no model was trained on images and blood values together.", ss["Italic"]))
    doc.build(el)
    return buf.getvalue()


# =====================  USER INTERFACE  =====================
CMAPS = {"Jet": cv2.COLORMAP_JET, "Turbo": cv2.COLORMAP_TURBO, "Inferno": cv2.COLORMAP_INFERNO, "Viridis": cv2.COLORMAP_VIRIDIS}

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## 🩺 DR Screening Suite")
    st.caption("Retina AI + Metabolic Risk")
    with st.spinner("Loading models..."):
        status = [("Classifier (EfficientNet-B0)",) + load_classifier()[1:],
                  ("Segmentation (U-Net)",) + load_seg()[1:],
                  ("Localization (ResNet)",) + load_loc()[1:]]
        flags = [load_classifier()[0] is not None, load_seg()[0] is not None, load_loc()[0] is not None]
    st.markdown("**Model status**")
    for (nm, msg), ok in zip([(s[0], s[1]) for s in status], flags):
        st.markdown(("✅ " if ok else "⚠️ ") + nm)
        if not ok:
            st.caption(msg)
    st.divider()

    src = st.radio("Image source", ["Upload", "Sample"], horizontal=True)
    img = None
    img_name = ""
    if src == "Upload":
        f = st.file_uploader("Fundus image", type=["png", "jpg", "jpeg"])
        if f is not None:
            img = Image.open(f).convert("RGB")
            img_name = f.name
    else:
        files = sorted([p for p in glob.glob(os.path.join(SAMPLES, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))])
        if files:
            pick = st.selectbox("Sample image", files, format_func=os.path.basename)
            img = Image.open(pick).convert("RGB")
            img_name = os.path.basename(pick)
        else:
            st.info("No images found in app/samples/")

    with st.expander("Display options", expanded=True):
        accent = st.selectbox("Colour theme", list(ACCENTS.keys()))
        view_mode = st.selectbox("Image enhancement", ["Original", "Contrast boost (CLAHE)", "Green channel (lesion contrast)"])
        layout = st.radio("Layout", ["Side by side", "Annotated only"], horizontal=True)
        alpha = st.slider("Grad-CAM opacity", 0.0, 1.0, 0.45, 0.05)
        cmap_name = st.selectbox("Heatmap colours", list(CMAPS.keys()))
        mask_alpha = st.slider("Lesion mask opacity", 0.1, 1.0, 0.5, 0.05)
        markers_on = st.checkbox("Show optic disc / fovea markers", True)
    with st.expander("Advanced"):
        cls_size = st.selectbox("Classifier input size", [224, 256, 288, 300], 0)
        seg_size = st.selectbox("Segmentation input size", [256, 384, 512], 0)
        seg_thr = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)
    with st.expander("Patient details"):
        p_name = st.text_input("Name / ID", "Demo patient")
        p_age = st.number_input("Age", 1, 110, 50)
    patient = {"name": p_name, "age": int(p_age)}

# ---------- hero ----------
_a1, _a2 = ACCENTS[accent]
st.markdown("<style>:root{--acc:%s;--acc2:%s}</style>" % (_a1, _a2), unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <h1>🩺 Diabetic Retinopathy + Metabolic Risk Assessment</h1>
  <p>Explainable retina screening (EfficientNet-B0, Grad-CAM, U-Net, landmark localization) combined with a metabolic risk score at decision level.</p>
  <span class="badge">Research prototype</span><span class="badge">Not a medical device</span><span class="badge">Explainable AI</span>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Models active", "%d / 3" % sum(flags))
k2.metric("APTOS accuracy", "84.6%")
k3.metric("APTOS QWK", "0.905")
k4.metric("IDRiD QWK (external)", "0.543")

# ---------- run pipeline ----------
res = None
disp = None
if img is not None:
    disp = resize_long(np.array(img), 640)
    key = hashlib.md5(disp.tobytes()).hexdigest() + "-%d-%d-%.2f" % (cls_size, seg_size, seg_thr)
    if st.session_state.get("key") != key:
        with st.spinner("Analyzing retina..."):
            st.session_state["res"] = analyze(disp, cls_size, seg_size, seg_thr)
        st.session_state["key"] = key
    res = st.session_state["res"]

tabs = st.tabs(["🔍 Retina Analysis", "🔥 Explainability", "🩸 Metabolic Risk", "📄 Combined Report", "📊 Model Performance"])

# ---------- tab 1 ----------
with tabs[0]:
    if res is None:
        st.info("Upload a fundus image or pick a sample from the sidebar to begin.")
    else:
        base = enhance(disp, view_mode)
        show = {}
        cols = st.columns(max(1, min(4, len(res["masks"]))))
        for i, nm in enumerate(res["masks"].keys()):
            show[nm] = cols[i % len(cols)].checkbox(nm, True, key="m_" + nm)
        annotated = compose(base, res["masks"], show, res["od"], res["fov"], markers_on, mask_alpha)
        if layout == "Side by side":
            c1, c2 = st.columns(2)
            c1.markdown("##### Original image")
            c1.image(base)
            c2.markdown("##### Lesions and landmarks")
            c2.image(annotated)
        else:
            st.markdown("##### Lesions and landmarks")
            st.image(annotated)
        st.caption("Lesions: " + res["seg_src"] + "  |  Landmarks: " + res["loc_src"])
        st.download_button("Download annotated image (PNG)", data=to_png_bytes(annotated).getvalue(),
                           file_name="annotated_retina.png", mime="image/png")
        if res["probs"] is None:
            st.error("Classifier not available. See the model status in the sidebar.")
        else:
            probs = res["probs"]
            grade = int(np.argmax(probs))
            st.markdown('<div class="card"><p class="note">Predicted grade</p><p class="grade">%d - %s</p><p>%s</p></div>'
                        % (grade, GRADES[grade], ADVICE[grade]), unsafe_allow_html=True)
            html = ""
            for i, p in enumerate(probs[:5]):
                html += ('<div class="row"><span>Grade %d - %s</span><span>%.1f%%</span></div>'
                         '<div class="bar-wrap"><div class="bar" style="width:%.1f%%"></div></div>') % (i, GRADES[i], 100 * p, 100 * p)
            st.markdown('<div class="card">' + html + "</div>", unsafe_allow_html=True)
        for e in res["errors"]:
            st.warning(e)

# ---------- tab 2 ----------
with tabs[1]:
    if res is None or res["cam"] is None:
        st.info("Grad-CAM needs an image and a loaded classifier.")
    else:
        cam_img = cam_overlay(enhance(disp, view_mode), res["cam"], alpha, CMAPS[cmap_name])
        a, b = st.columns(2)
        a.markdown("##### Original")
        a.image(enhance(disp, view_mode))
        b.markdown("##### Grad-CAM (opacity %.2f)" % alpha)
        b.image(cam_img)
        st.caption("Warm regions are the areas that pushed the classifier towards its predicted grade. "
                   "Use the opacity slider in the sidebar to blend the heatmap.")
    grid = os.path.join(OUT, "gradcam_samples", "grid.png")
    if os.path.exists(grid):
        with st.expander("Grad-CAM examples, one per grade (from training)"):
            st.image(grid)

# ---------- tab 3 ----------
with tabs[2]:
    with st.form("risk_form"):
        r1, r2, r3 = st.columns(3)
        hba = r1.slider("HbA1c (%)", 4.0, 14.0, 7.0, 0.1)
        sbp = r2.slider("Systolic BP (mmHg)", 90, 200, 120)
        dur = r3.slider("Diabetes duration (years)", 0, 40, 8)
        go = st.form_submit_button("Calculate Metabolic Risk")
    if go:
        st.session_state["risk"] = dict(hba=hba, sbp=sbp, dur=dur, **metabolic(hba, sbp, dur))
    r = st.session_state.get("risk")
    if r:
        st.markdown('<div class="card"><p class="note">Metabolic risk level (score %d / 9)</p>'
                    '<span class="pill" style="background:%s">%s</span></div>' % (r["score"], LEVEL_COL[r["idx"]], r["level"]),
                    unsafe_allow_html=True)
        st.progress(r["score"] / 9.0)
    else:
        st.info("Set the sliders and press the button.")

# ---------- tab 4 ----------
with tabs[3]:
    r = st.session_state.get("risk")
    if res is None or res["probs"] is None:
        st.info("Needs an analysed image first.")
    elif not r:
        st.info("Calculate the metabolic risk in the previous tab first.")
    else:
        grade = int(np.argmax(res["probs"]))
        fi = fuse(grade, r["idx"])
        x1, x2, x3 = st.columns(3)
        x1.metric("Retina grade", "%d - %s" % (grade, GRADES[grade]))
        x2.metric("Metabolic level", r["level"])
        x3.metric("Combined risk", LEVELS[fi])
        st.markdown('<div class="card"><span class="pill" style="background:%s">%s</span><p style="margin-top:10px">%s</p></div>'
                    % (LEVEL_COL[fi], LEVELS[fi], COMBINED_ADVICE[fi]), unsafe_allow_html=True)
        st.caption("Decision-level fusion: the two scores are computed separately and combined afterwards.")
        try:
            cam_for_pdf = cam_overlay(disp, res["cam"], alpha, CMAPS[cmap_name])
            pdf = build_pdf(patient, grade, res["probs"], r, fi, disp, cam_for_pdf, res["seg_src"])
            st.download_button("Download Full Clinical PDF Report", data=pdf,
                               file_name="DR_report_%s.pdf" % datetime.date.today().isoformat(), mime="application/pdf")
        except Exception as e:
            st.error("PDF could not be built: " + str(e)[:200])

# ---------- tab 5 ----------
with tabs[4]:
    st.markdown("##### Results (from Person B's evaluation)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("APTOS accuracy", "84.6%")
    m2.metric("APTOS QWK", "0.905")
    m3.metric("IDRiD accuracy", "38.8%")
    m4.metric("IDRiD QWK", "0.543")
    st.markdown('<div class="card"><b>Finding: domain shift.</b> The model was trained on APTOS and tested on IDRiD, '
                'which uses a different camera and population. The drop in performance shows why local validation '
                'is needed before any clinical use. The weakest grades are Severe and Proliferative, which have few training examples.</div>',
                unsafe_allow_html=True)
    ca, cb = st.columns(2)
    for col, fn in ((ca, "confusion_matrix.png"), (cb, "training_curves.png")):
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            col.image(p)
    mp = os.path.join(MODELS, "metrics.json")
    if os.path.exists(mp):
        with st.expander("Raw metrics.json"):
            try:
                st.json(json.load(open(mp)))
            except Exception as e:
                st.warning("Could not read metrics.json: " + str(e)[:150])

st.caption("Research prototype for education. Not a medical device. Not for clinical use.")
