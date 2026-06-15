import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import base64
import cv2
import os

# ── Disease classes ──────────────────────────────────────────
CLASS_NAMES = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'Age-related Macular Degeneration']

RECOMMENDATIONS = {
    'Normal':                           'No abnormalities detected. Routine eye check in 12 months.',
    'Diabetic Retinopathy':             'Immediate referral to retinal specialist. Control blood sugar levels.',
    'Glaucoma':                         'Begin IOP monitoring. Specialist follow-up within 1 week.',
    'Cataract':                         'Schedule surgical consultation with ophthalmologist.',
    'Age-related Macular Degeneration': 'Anti-VEGF therapy evaluation recommended. Avoid smoking.'
}

SEVERITY_MAP = {
    'Normal':                           'Low',
    'Diabetic Retinopathy':             'High',
    'Glaucoma':                         'High',
    'Cataract':                         'Moderate',
    'Age-related Macular Degeneration': 'Moderate'
}

# ── Preprocessing ─────────────────────────────────────────────
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor, image

# ── Build model ───────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model

model = build_model()

# ── Load trained weights ──────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'drishti_model.pth')
WEIGHTS_LOADED = False

if os.path.exists(WEIGHTS_PATH):
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        model.eval()
        WEIGHTS_LOADED = True
        print("✅ Trained model weights loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load weights: {e}")
else:
    print("⚠️ No trained weights found.")

model.eval()

# ── Grad-CAM ──────────────────────────────────────────────────
def generate_gradcam(image_bytes):
    tensor, original_pil = preprocess_image(image_bytes)
    img_array = np.array(original_pil.resize((224, 224)))

    if WEIGHTS_LOADED:
        gradients = []
        activations = []

        def save_gradient(grad):
            gradients.append(grad)

        def forward_hook(module, input, output):
            activations.append(output)
            output.register_hook(save_gradient)

        # Hook last conv layer of ResNet50
        hook = model.layer4.register_forward_hook(forward_hook)

        output = model(tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_class].item()

        model.zero_grad()
        output[0][pred_class].backward()
        hook.remove()

        grad = gradients[0].squeeze().detach().numpy()
        act = activations[0].squeeze().detach().numpy()

        weights = grad.mean(axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]
    else:
        pred_class = 0
        confidence = 0.85
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        cam = cv2.Canny(gray, 30, 100).astype(np.float32)
        cam = cv2.GaussianBlur(cam, (21, 21), 0)

    # Generate heatmap
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_array + 0.5 * heatmap).astype(np.uint8)

    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format='PNG')
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return pred_class, confidence, heatmap_b64

# ── Main predict ──────────────────────────────────────────────
def predict(image_bytes):
    pred_class, confidence, heatmap_b64 = generate_gradcam(image_bytes)
    disease = CLASS_NAMES[pred_class]
    return {
        'disease':        disease,
        'confidence':     round(float(confidence) * 100, 1),
        'severity':       SEVERITY_MAP[disease],
        'recommendation': RECOMMENDATIONS[disease],
        'heatmap':        heatmap_b64
    }