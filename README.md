# rPPG 無接觸式生理數據輔助感測 PoC

這是一套可直接從一般 RGB Webcam 啟動的 rPPG Web PoC：

- React / Vite 前端：鏡頭預覽、臉框、HR / RR / HRV / SQI、rPPG waveform
- FastAPI / WebSocket 後端：接收壓縮影像、臉部 ROI、RGB temporal signal、POS / CHROM、頻譜分析
- 預設 **不需要單獨下載 rPPG 模型權重**
- 預設也 **不需要額外下載人臉模型**：第一版使用 OpenCV Python wheel 內附的 Haar cascade
- `backend/models/` 已預留未來接 PhysNet / PhysFormer / EfficientPhys / PhysMamba checkpoint 的位置

> 本專案僅供研究、展示與輔助感測，不是醫療器材，不應用於疾病診斷、用藥、保險、緊急醫療或其他高風險決策。

## 架構

```text
Browser / React
  getUserMedia (1280x720 / ideal 30 fps)
        |
        | JPEG frames ~12.5 fps over WebSocket
        v
FastAPI /ws/rppg
        |
        +-- OpenCV face detection
        +-- forehead + cheek ROIs
        +-- skin/exposure filtering
        +-- RGB temporal buffer (24 sec)
        +-- POS / CHROM
        +-- band-pass 0.7-3.0 Hz
        +-- Welch spectrum -> HR
        +-- envelope spectrum -> rough RR
        +-- pulse peaks -> rough HRV RMSSD
        +-- SQI = spectral confidence + motion/exposure penalties
        |
        v
JSON vitals -> React dashboard
```

## 為什麼第一版不需要模型？

POS 與 CHROM 是 classical / unsupervised rPPG 方法，不是需要 checkpoint 的 neural network。第一版直接從臉部 ROI 的 RGB 時序訊號運算，因此不需要 `.pth` / `.onnx` / `.task` 等 rPPG 模型權重。

若後續切換到 PhysNet / PhysFormer / EfficientPhys / PhysMamba 等 supervised neural methods，才需要下載對應 checkpoint，而且需要逐一確認程式碼與模型權重授權。

## Windows 快速啟動

### 需求

- Python 3.11 建議
- Node.js 20.19+ 或 22.12+（Vite 8 要求）
- Chrome / Edge
- Webcam：720p/30fps 可用；1080p/30fps 以上較佳

### 1. 啟動後端

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

確認：瀏覽器開 `http://localhost:8000/api/health`

### 2. 啟動前端

另開一個 PowerShell：

```powershell
cd frontend
npm install
npm run dev
```

打開：`http://localhost:5173`

### 3. 操作

1. 按「啟動鏡頭」並允許瀏覽器 camera permission。
2. 臉部保持正面，建議距離 50–80 cm。
3. 固定室內光源，避免窗戶背光與燈光頻繁變化。
4. 前 8 秒為訊號累積；HR 之後才開始顯示。
5. RR 需較長時窗，PoC 約 18 秒後才嘗試顯示。
6. 可切換 POS / CHROM；切換會清空 buffer 重新量測。

## Docker Compose

```bash
docker compose up --build
```

前端：http://localhost:5173  
後端：http://localhost:8000

## API

### Health

`GET /api/health`

### WebSocket

`WS /ws/rppg`

設定 engine：

```json
{"type":"config","engine":"pos"}
```

傳 frame：

```json
{
  "type": "frame",
  "timestamp": 1786610000.123,
  "image": "data:image/jpeg;base64,..."
}
```

回傳：

```json
{
  "type": "vitals",
  "status": "good",
  "hr": 72.4,
  "rr": 15.1,
  "hrv_rmssd": 38.2,
  "signal_quality": 0.78,
  "waveform": [0.1, 0.3, 0.2],
  "bbox": {"x":0.3,"y":0.18,"w":0.28,"h":0.5},
  "engine": "pos",
  "fps": 12.5,
  "window_seconds": 19.2,
  "motion": 0.04,
  "exposure": 0.88
}
```

## 設備建議

| 項目 | 最低 PoC | 建議 Demo |
|---|---|---|
| Camera | 720p / 30 fps | 1080p / 30–60 fps |
| CPU | Core i5 / Ryzen 5 | Core i7 / Ryzen 7 |
| GPU | 不需要 | 深度學習版才需要 NVIDIA GPU |
| RAM | 8 GB | 16 GB |
| 光源 | 穩定室內燈 | 正面柔光 LED |
| 距離 | 約 40–100 cm | 約 50–80 cm |

## 已知限制

- 一般 webcam 的 auto exposure / white balance 會干擾微小的 RGB 變化。
- 頭部移動、說話幅度大、表情、手遮臉、眼鏡反光與背景閃爍都會降低 SQI。
- HRV 與 RR 由 video-rPPG 估算，在短時窗與一般 webcam 下僅適合作為研究 / demo 指標。
- SpO2 與血壓不應直接用這個 RGB Webcam PoC 宣稱為可靠量測值。
- 正式驗證建議同步使用指夾式 PPG / ECG / 呼吸帶作 ground truth，收集不同膚色、光線、攝影機與 motion 條件。

## 下一階段建議

1. OpenCV Haar -> MediaPipe Face Landmarker，取得更穩定的 forehead / cheeks polygons。
2. 支援 raw / MJPEG / WebRTC，降低 JPEG 傳輸與壓縮造成的色彩擾動。
3. Camera control：鎖 exposure / white balance / focus（硬體支援時）。
4. Ground-truth recorder：同步儲存 webcam RGB trace + reference PPG/ECG。
5. Model adapter：PhysNet / EfficientPhys / PhysFormer。
6. calibration dashboard：MAE、RMSE、Pearson r、Bland–Altman。

## 專案結構

```text
rppg-web-poc/
├─ backend/
│  ├─ app/
│  │  ├─ engines/classical.py
│  │  ├─ services/vision.py
│  │  └─ main.py
│  ├─ models/README.md
│  ├─ tests/test_classical.py
│  └─ requirements.txt
├─ frontend/
│  ├─ src/
│  ├─ package.json
│  └─ .env.example
├─ docker-compose.yml
└─ README.md
```
