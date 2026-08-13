import { useEffect, useRef, useState } from 'react'
import { Activity, Camera, HeartPulse, Radio, ShieldCheck } from 'lucide-react'
import MetricCard from './components/MetricCard'
import Waveform from './components/Waveform'

const WS_URL = import.meta.env.VITE_RPPG_WS || 'ws://localhost:8000/ws/rppg'

function qualityText(q, status) {
  if (status === 'no_face') return ['未偵測到臉部', 'bad']
  if (status === 'measuring') return ['量測準備中', 'wait']
  if (q >= 0.7) return ['訊號良好', 'good']
  if (q >= 0.45) return ['訊號可用', 'mid']
  return ['訊號偏弱', 'bad']
}

export default function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const socketRef = useRef(null)
  const timerRef = useRef(null)
  const [cameraOn, setCameraOn] = useState(false)
  const [connected, setConnected] = useState(false)
  const [engine, setEngine] = useState('pos')
  const [vitals, setVitals] = useState({
    status: 'idle', hr: null, rr: null, hrv_rmssd: null,
    signal_quality: 0, waveform: [], bbox: null, window_seconds: 0,
    fps: null, motion: null, exposure: null,
  })
  const [error, setError] = useState('')

  useEffect(() => () => stopCamera(), [])

  const connectSocket = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return socketRef.current
    const ws = new WebSocket(WS_URL)
    socketRef.current = ws
    ws.onopen = () => {
      setConnected(true)
      ws.send(JSON.stringify({ type: 'config', engine }))
    }
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setError('無法連線到 rPPG 後端，請確認 FastAPI 已在 8000 port 啟動。')
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      if (msg.type === 'vitals') setVitals(msg)
      if (msg.type === 'error') setError(msg.message || '後端發生錯誤')
    }
    return ws
  }

  const startCamera = async () => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, min: 15 },
          facingMode: 'user',
        },
        audio: false,
      })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      setCameraOn(true)
      connectSocket()
      timerRef.current = window.setInterval(sendFrame, 80) // ~12.5 fps transport
    } catch (e) {
      setError(`鏡頭啟動失敗：${e.message}`)
    }
  }

  const stopCamera = () => {
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = null
    const stream = videoRef.current?.srcObject
    stream?.getTracks?.().forEach(t => t.stop())
    if (videoRef.current) videoRef.current.srcObject = null
    socketRef.current?.close()
    socketRef.current = null
    setCameraOn(false)
    setConnected(false)
    setVitals({ status: 'idle', hr: null, rr: null, hrv_rmssd: null, signal_quality: 0, waveform: [], bbox: null, window_seconds: 0 })
  }

  const sendFrame = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const ws = socketRef.current
    if (!video || !canvas || ws?.readyState !== WebSocket.OPEN || video.readyState < 2) return
    if (ws.bufferedAmount > 750_000) return
    const w = 480
    const h = Math.round(w * (video.videoHeight || 720) / (video.videoWidth || 1280))
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d', { alpha: false })
    ctx.drawImage(video, 0, 0, w, h)
    const image = canvas.toDataURL('image/jpeg', 0.72)
    ws.send(JSON.stringify({ type: 'frame', timestamp: Date.now() / 1000, image }))
  }

  const changeEngine = (next) => {
    setEngine(next)
    setVitals(v => ({ ...v, hr: null, rr: null, hrv_rmssd: null, waveform: [], window_seconds: 0 }))
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'config', engine: next }))
    }
  }

  const [qLabel, qClass] = qualityText(vitals.signal_quality || 0, vitals.status)
  const qPct = Math.round((vitals.signal_quality || 0) * 100)
  const progress = Math.min(100, Math.round((vitals.window_seconds || 0) / 8 * 100))

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow"><Radio size={15} /> rPPG REAL-TIME POC</div>
          <h1>無接觸式生理數據輔助感測</h1>
          <p>RGB Webcam · POS / CHROM · FastAPI WebSocket · React</p>
        </div>
        <div className={`connection ${connected ? 'online' : ''}`}>
          <span className="dot" /> {connected ? 'Backend Connected' : 'Backend Offline'}
        </div>
      </header>

      <section className="layout">
        <div className="camera-panel panel">
          <div className="panel-head">
            <div><Camera size={18} /> 即時影像</div>
            <div className="engine-switch">
              <button className={engine === 'pos' ? 'active' : ''} onClick={() => changeEngine('pos')}>POS</button>
              <button className={engine === 'chrom' ? 'active' : ''} onClick={() => changeEngine('chrom')}>CHROM</button>
            </div>
          </div>

          <div className="video-wrap">
            <video ref={videoRef} muted playsInline />
            {!cameraOn && <div className="camera-placeholder"><HeartPulse size={48} /><span>啟動鏡頭開始量測</span></div>}
            {cameraOn && vitals.bbox && (
              <div className="face-box" style={{
                left: `${(1 - vitals.bbox.x - vitals.bbox.w) * 100}%`, top: `${vitals.bbox.y * 100}%`,
                width: `${vitals.bbox.w * 100}%`, height: `${vitals.bbox.h * 100}%`,
              }}><span>FACE ROI</span></div>
            )}
            {cameraOn && <div className="scan-line" />}
          </div>
          <canvas ref={canvasRef} className="capture-canvas" />

          <div className="camera-controls">
            <button className="primary" onClick={cameraOn ? stopCamera : startCamera}>{cameraOn ? '停止量測' : '啟動鏡頭'}</button>
            <div className="capture-note">建議臉部距離 50–80 cm、光線固定、保持不動</div>
          </div>

          {cameraOn && (
            <div className="quality-box">
              <div className="quality-row">
                <strong className={qClass}>{qLabel}</strong>
                <span>{vitals.status === 'measuring' ? `${Math.round(vitals.window_seconds || 0)} / 8 秒` : `${qPct}% SQI`}</span>
              </div>
              <div className="progress"><span style={{ width: `${vitals.status === 'measuring' ? progress : qPct}%` }} /></div>
              <div className="quality-tips">
                <span>Motion {vitals.motion == null ? '—' : Math.round(vitals.motion * 100) + '%'}</span>
                <span>Exposure {vitals.exposure == null ? '—' : Math.round(vitals.exposure * 100) + '%'}</span>
                <span>FPS {vitals.fps ?? '—'}</span>
              </div>
            </div>
          )}
          {error && <div className="error-box">{error}</div>}
        </div>

        <div className="right-column">
          <div className="metrics-grid">
            <MetricCard label="心率 HR" value={vitals.hr} unit="BPM" hint="主要展示指標" accent />
            <MetricCard label="呼吸率 RR" value={vitals.rr} unit="/min" hint="需約 18 秒訊號" />
            <MetricCard label="HRV" value={vitals.hrv_rmssd} unit="ms" hint="RMSSD，實驗性估計" />
            <MetricCard label="訊號品質 SQI" value={cameraOn ? qPct : null} unit="%" hint="含頻譜、動作、曝光" />
          </div>

          <div className="panel waveform-panel">
            <div className="panel-head">
              <div><Activity size={18} /> rPPG Waveform</div>
              <span className="mono">ENGINE: {engine.toUpperCase()}</span>
            </div>
            <Waveform values={vitals.waveform} />
          </div>

          <div className="panel disclaimer">
            <ShieldCheck size={20} />
            <div><strong>輔助感測，不是醫療診斷</strong><p>本 PoC 以一般 RGB Webcam 估算生理訊號；動作、照明、膚色、壓縮、鏡頭自動曝光等都可能影響結果。請勿作為疾病診斷、用藥或緊急醫療判斷依據。</p></div>
          </div>
        </div>
      </section>
    </main>
  )
}
