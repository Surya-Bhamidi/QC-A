"""
app.py - Flask Backend for Interactive QViT Research & Viva Demonstration Dashboard
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
import json
import base64
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_from_directory

from src.models.classical_vit import ClassicalViT
from src.models.qvit import QViT
from src.quantum.swap_test import QuantumSwapTestModule

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=STATIC_DIR)

# Device & Model setup
DEVICE = torch.device("cpu")
print("Loading trained checkpoints for Dashboard...")

# 1. Classical ViT
classical_model = ClassicalViT(
    image_size=128, patch_size=32, in_channels=3, num_classes=2,
    embed_dim=32, depth=2, num_heads=2, mlp_dim=64
).to(DEVICE)
classical_ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "classical_vit_best.pt")
if os.path.exists(classical_ckpt_path):
    ckpt = torch.load(classical_ckpt_path, map_location=DEVICE)
    classical_model.load_state_dict(ckpt["model_state_dict"])
    classical_model.eval()
    print("Loaded Classical ViT checkpoint.")

# 2. QViT Hybrid
qvit_model = QViT(
    image_size=128, patch_size=32, in_channels=3, num_classes=2,
    embed_dim=32, n_qubits=4, num_heads=2, depth=2, mlp_dim=64
).to(DEVICE)
qvit_ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "qvit_best.pt")
if os.path.exists(qvit_ckpt_path):
    ckpt = torch.load(qvit_ckpt_path, map_location=DEVICE)
    qvit_model.load_state_dict(ckpt["model_state_dict"])
    qvit_model.eval()
    print("Loaded QViT checkpoint.")

# Load Test Dataset Samples
NPZ_PATH = os.path.join(PROJECT_ROOT, "data", "pneumoniamnist.npz")
SAMPLES = []
if os.path.exists(NPZ_PATH):
    data = np.load(NPZ_PATH)
    test_imgs = data["test_images"]
    test_lbls = data["test_labels"].flatten()

    # Select 4 Normal and 4 Pneumonia samples
    normal_indices = np.where(test_lbls == 0)[0][:4]
    pneumonia_indices = np.where(test_lbls == 1)[0][:4]

    selected_indices = list(normal_indices) + list(pneumonia_indices)
    for idx in selected_indices:
        img_arr = test_imgs[idx]
        lbl = int(test_lbls[idx])
        label_name = "Normal" if lbl == 0 else "Pneumonia"
        
        # Convert to Base64 thumbnail
        pil_img = Image.fromarray(img_arr).convert("RGB").resize((128, 128))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        SAMPLES.append({
            "id": int(idx),
            "label": label_name,
            "label_idx": lbl,
            "thumbnail_b64": f"data:image/png;base64,{b64_str}",
            "description": f"Patient #{idx} - Ground Truth: {label_name}",
        })
    print(f"Loaded {len(SAMPLES)} test cases for interactive demonstration.")

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def generate_heatmap_overlay(pil_img: Image.Image, attn_grid_4x4: torch.Tensor, colormap_name: str = "magma") -> str:
    """Interpolates 4x4 attention grid to 128x128 and overlays onto the original X-ray."""
    attn_upsampled = F.interpolate(
        attn_grid_4x4.unsqueeze(0).unsqueeze(0),
        size=(128, 128),
        mode="bilinear",
        align_corners=False
    )[0, 0].cpu().numpy()

    # Normalize between 0 and 1
    attn_norm = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)

    # Convert original image to grayscale float
    img_gray = np.array(pil_img.convert("L").resize((128, 128))) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap_name)
    colored_heatmap = cmap(attn_norm)[:, :, :3]

    # Blend
    blended = 0.55 * np.repeat(img_gray[:, :, None], 3, axis=-1) + 0.45 * colored_heatmap
    blended = np.clip(blended, 0.0, 1.0)
    blended_uint8 = (blended * 255).astype(np.uint8)

    out_img = Image.fromarray(blended_uint8)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify({"samples": SAMPLES})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}
    sample_id = data.get("sample_id")
    image_b64 = data.get("image_b64")
    noise_prob = float(data.get("noise_prob", 0.0))

    if image_b64:
        # Decode custom uploaded image
        try:
            raw_bytes = base64.b64decode(image_b64.split(",")[-1])
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((128, 128))
            ground_truth = "Unknown (User Upload)"
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {str(e)}"}), 400
    elif sample_id is not None:
        idx = int(sample_id)
        img_arr = np.load(NPZ_PATH)["test_images"][idx]
        lbl = int(np.load(NPZ_PATH)["test_labels"].flatten()[idx])
        pil_img = Image.fromarray(img_arr).convert("RGB").resize((128, 128))
        ground_truth = "Normal" if lbl == 0 else "Pneumonia"
    else:
        # Default to first sample
        img_arr = np.load(NPZ_PATH)["test_images"][0]
        pil_img = Image.fromarray(img_arr).convert("RGB").resize((128, 128))
        ground_truth = "Normal"

    # Preprocess
    tensor_in = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    # 1. Classical ViT Inference
    t0 = time.time()
    with torch.no_grad():
        c_logits = classical_model(tensor_in)
        c_probs = F.softmax(c_logits, dim=-1)[0].cpu().numpy()
        c_pred = int(c_probs.argmax())
    c_latency = (time.time() - t0) * 1000

    # 2. QViT Inference with Noise Setting
    qvit_model.set_noise(noise_prob)
    t0 = time.time()
    with torch.no_grad():
        q_logits, q_attns = qvit_model(tensor_in, return_attn=True)
        q_probs = F.softmax(q_logits, dim=-1)[0].cpu().numpy()
        q_pred = int(q_probs.argmax())
        
        # Last layer attention map
        last_attn = q_attns[-1][0]  # (17, 17)
        # Class token to patches
        cls_attn = last_attn[0, 1:].view(4, 4)
    q_latency = (time.time() - t0) * 1000

    # Overlays
    q_overlay_b64 = generate_heatmap_overlay(pil_img, cls_attn, colormap_name="plasma")
    
    # Original raw image b64
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    raw_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    return jsonify({
        "ground_truth": ground_truth,
        "raw_image": raw_b64,
        "noise_prob": noise_prob,
        "classical": {
            "prediction": "Pneumonia" if c_pred == 1 else "Normal",
            "pred_idx": c_pred,
            "prob_normal": float(c_probs[0]),
            "prob_pneumonia": float(c_probs[1]),
            "confidence": float(c_probs[c_pred]),
            "latency_ms": round(c_latency, 2),
        },
        "quantum": {
            "prediction": "Pneumonia" if q_pred == 1 else "Normal",
            "pred_idx": q_pred,
            "prob_normal": float(q_probs[0]),
            "prob_pneumonia": float(q_probs[1]),
            "confidence": float(q_probs[q_pred]),
            "latency_ms": round(q_latency, 2),
            "overlay_b64": q_overlay_b64,
            "attention_grid": cls_attn.tolist(),
        }
    })


@app.route("/api/swap_test_calc", methods=["POST"])
def swap_test_calc():
    """Calculates SWAP test circuit state evolution for user-provided Query and Key vectors."""
    data = request.json or {}
    q_vec = data.get("q", [0.5, -0.2, 0.8, -0.5])
    k_vec = data.get("k", [0.4, -0.3, 0.7, -0.4])

    q_t = torch.tensor([q_vec], dtype=torch.float32)
    k_t = torch.tensor([k_vec], dtype=torch.float32)

    # Angle embedding: theta = (pi/2) * (tanh(x) + 1)
    q_angles = ((torch.tanh(q_t) + 1.0) * (torch.pi / 2.0))[0].tolist()
    k_angles = ((torch.tanh(k_t) + 1.0) * (torch.pi / 2.0))[0].tolist()

    # Step-by-step fidelity per qubit
    diffs = [(q_a - k_a) / 2.0 for q_a, k_a in zip(q_angles, k_angles)]
    wire_fidelities = [float(np.cos(d)**2) for d in diffs]
    total_fidelity = float(np.prod(wire_fidelities))

    return jsonify({
        "query_vector": q_vec,
        "key_vector": k_vec,
        "query_angles_rad": [round(a, 4) for a in q_angles],
        "key_angles_rad": [round(a, 4) for a in k_angles],
        "wire_fidelities": [round(f, 4) for f in wire_fidelities],
        "state_fidelity": round(total_fidelity, 5),
        "steps": [
            {"step": 1, "name": "State Preparation", "desc": "Continuous features are mapped to rotation angles theta in [0, pi] via RY gates on Register Q (wires 1-4) and Register K (wires 5-8)."},
            {"step": 2, "name": "Ancilla Superposition", "desc": "Hadamard gate applied to Ancilla wire 0: H|0>_a = (|0>_a + |1>_a) / sqrt(2)."},
            {"step": 3, "name": "Controlled-SWAP (Fredkin)", "desc": "Controlled on Ancilla wire 0, pairwise swap Q_i <-> K_i occurs if ancilla is |1>."},
            {"step": 4, "name": "Ancilla Interference", "desc": "Second Hadamard gate converts relative phase difference into constructive/destructive probability amplitude."},
            {"step": 5, "name": "Pauli-Z Measurement", "desc": f"Measuring Pauli-Z on Ancilla yields <Z_a> = P(|0>) - P(|1>) = {total_fidelity:.5f} (Exact State Overlap)."},
        ]
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    c_metrics_path = os.path.join(PROJECT_ROOT, "results", "classical_vit_metrics.json")
    q_metrics_path = os.path.join(PROJECT_ROOT, "results", "qvit_metrics.json")
    noise_path = os.path.join(PROJECT_ROOT, "results", "noise_resilience.json")

    out = {}
    if os.path.exists(c_metrics_path):
        with open(c_metrics_path) as f:
            out["classical"] = json.load(f)
    if os.path.exists(q_metrics_path):
        with open(q_metrics_path) as f:
            out["quantum"] = json.load(f)
    if os.path.exists(noise_path):
        with open(noise_path) as f:
            out["noise_sweep"] = json.load(f)

    return jsonify(out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n=======================================================")
    print(f" Quantum Vision Transformer (QViT) Dashboard Active! ")
    print(f" Serving live at: http://localhost:{port}            ")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
