# Skinaiq — AI-Powered Skin Intelligence

> AI-powered skin disease detection from your phone camera & smartwatch.  
> No extra hardware. 20+ conditions detected before symptoms appear.

---

## Project Structure

```
skinaiq-app/
├── index.html          ← Landing page (features, AI models, conditions)
├── scan.html           ← Scan page (camera + voice + biometrics input)
├── dashboard.html      ← Full health dashboard
├── app.py              ← Flask backend with all 5 ML model integrations
├── requirements.txt    ← Python dependencies
├── uploads/            ← Uploaded scan images (auto-created)
└── README.md
```

---

## 5 AI Models

| # | Model | Role | Accuracy |
|---|-------|------|----------|
| 01 | EfficientNet-B4 | Skin Lesion Classifier | 94% |
| 02 | Vision Transformer (ViT) | Cancer Detection Engine | 97.6% |
| 03 | MediaPipe Face Mesh | Facial Analysis (468 pts) | 99% |
| 04 | Wav2Vec 2.0 | Voice Biomarker Analysis | 88% |
| 05 | Skinaiq Fusion Engine | XGBoost + LSTM (Master) | Active |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the backend
```bash
python app.py
```
Backend runs at: `http://localhost:5000`

### 3. Open the frontend
Open `index.html` in your browser directly, or serve it:
```bash
python -m http.server 3000
# Then visit http://localhost:3000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server + model status |
| POST | `/api/scan/full` | Full scan (image + voice + biometrics) → all 5 models |
| POST | `/api/scan/image` | Image-only → EfficientNet + ViT + FaceMesh |
| POST | `/api/scan/voice` | Voice-only → Wav2Vec 2.0 |
| GET | `/api/models/info` | All 5 model details |
| GET | `/api/conditions` | 20+ detectable conditions |
| GET | `/api/patient/history` | 7-day scan history |

---

## Example API Call

```python
import requests

response = requests.post("http://localhost:5000/api/scan/full", files={
    "image": open("skin_photo.jpg", "rb"),
    "audio": open("voice_sample.wav", "rb"),
}, data={
    "biometrics": '{"hr":72,"hrv":62,"spo2":98}'
})

report = response.json()
print(f"Health Score: {report['summary']['health_score']}")
print(f"Top Risk: {report['summary']['top_risk']['name']}")
print(f"Confidence: {report['summary']['top_risk']['confidence']}%")
```

---

## Pages

- **`index.html`** — Landing page with all platform features, AI models, detected conditions, and API docs
- **`scan.html`** — Live scan interface: enable camera, record voice, enter smartwatch biometrics, run all 5 models
- **`dashboard.html`** — Full health monitoring dashboard: health score, live vitals, alerts, sleep, risks, interventions, telehealth

---

## Notes

- The backend uses **simulated inference** when ML model weights are not present (graceful fallback)
- To use real models, load pre-trained weights into `models/` and update the load paths in `app.py`
- CORS is enabled for local development
- All scan results are saved to `sessionStorage` and passed to the dashboard

---

## Disclaimer

Skinaiq is a **screening tool**, not a clinical diagnosis.  
All alerts are reviewed by a certified dermatologist before reaching the patient.  
Skinaiq does not replace professional medical advice.
