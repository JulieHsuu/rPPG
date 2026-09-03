import { useEffect, useRef, useState } from 'react'
import { Camera, HeartPulse, History, Radio, ShieldCheck } from 'lucide-react'
import MetricCard from './components/MetricCard'
import WellnessRadar from './components/WellnessRadar'
import ResultsView from './components/ResultsView'
import VitalsBody from './components/VitalsBody'
import { createLocalRppg } from './lib/localRppg'
import HistoryView from './components/HistoryView'

const WS_URL = import.meta.env.VITE_RPPG_WS || 'ws://localhost:8000/ws/rppg'
const HISTORY_KEY = 'rppg-measurement-history-v1'

function qualityText(q, status) {
  if (status === 'no_face') return ['未偵測到臉部', 'bad']
  if (status === 'paused') return ['臉部遺失，進度已暫停', 'bad']
  if (status === 'complete') return ['量測完成', 'good']
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
  const frameCallbackRef = useRef(null)
  const lastFrameTimeRef = useRef(-1)
  const captureActiveRef = useRef(false)
  const localAnalyzerRef = useRef(null)
  const savedResultRef = useRef(false)
  // Local measurements should work from the Vite development server without a
  // separately running FastAPI process. Opt into the backend explicitly when
  // testing the Python analyser.
  const useLocalAnalysis = import.meta.env.VITE_USE_BACKEND !== 'true'
  const [cameraOn, setCameraOn] = useState(false)
  const [connected, setConnected] = useState(false)
  const [engine, setEngine] = useState('pos')
  const [vitals, setVitals] = useState({
    status: 'idle', hr: null, rr: null, hrv_rmssd: null,
    signal_quality: 0, waveform: [], bbox: null, window_seconds: 0,
    fps: null, motion: null, exposure: null,
    roi_quality: null,
    measurement_progress: 0, spo2: null, stress_index: null, wellness: null,
  })
  const [error, setError] = useState('')
  const [showHistory, setShowHistory] = useState(false)
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]') }
    catch { return [] }
  })

  useEffect(() => () => stopCamera(), [])

  useEffect(() => {
    if (vitals.status !== 'complete' || savedResultRef.current) return
    savedResultRef.current = true
    const wellness = vitals.wellness || {}
    const record = {
      id: crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`,
      measured_at: new Date().toISOString(), engine: vitals.engine?.toUpperCase?.() || '',
      sqi: Math.round((vitals.signal_quality || 0) * 100), seconds: Math.round(vitals.window_seconds || 0),
      hr: vitals.hr, hrv: vitals.hrv_rmssd, spo2: vitals.spo2, rr: vitals.rr,
      sbp: vitals.bp_systolic, dbp: vitals.bp_diastolic, stress: vitals.stress_index,
      activity: wellness.activity, sleep: wellness.sleep, equilibrium: wellness.equilibrium,
      metabolism: wellness.metabolism, health: wellness.health, relaxation: wellness.relaxation,
    }
    setHistory(current => {
      const next = [record, ...current].slice(0, 200)
      localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
      return next
    })
  }, [vitals])

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
      if (msg.type === 'vitals') {
        setVitals(msg)
        if (msg.status === 'complete') finishCapture()
      }
      if (msg.type === 'error') setError(msg.message || '後端發生錯誤')
    }
    return ws
  }

  const startCamera = async () => {
    setError('')
    savedResultRef.current = false
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
      captureActiveRef.current = true
      if (useLocalAnalysis) {
        localAnalyzerRef.current = createLocalRppg(msg => {
          setVitals(msg)
          if (msg.status === 'complete') finishCapture()
        })
      } else connectSocket()
      lastFrameTimeRef.current = -1
      if (typeof videoRef.current.requestVideoFrameCallback === 'function') {
        const captureFrame = (_now, metadata) => {
          if (metadata.mediaTime - lastFrameTimeRef.current >= 0.04) {
            lastFrameTimeRef.current = metadata.mediaTime
            sendFrame(metadata.mediaTime)
          }
          if (captureActiveRef.current) {
            frameCallbackRef.current = videoRef.current?.requestVideoFrameCallback(captureFrame)
          }
        }
        frameCallbackRef.current = videoRef.current.requestVideoFrameCallback(captureFrame)
      } else {
        timerRef.current = window.setInterval(() => sendFrame(performance.now() / 1000), 40)
      }
    } catch (e) {
      setError(`鏡頭啟動失敗：${e.message}`)
    }
  }

  const stopCamera = () => {
    if (timerRef.current) window.clearInterval(timerRef.current)
    captureActiveRef.current = false
    timerRef.current = null
    if (frameCallbackRef.current != null && videoRef.current?.cancelVideoFrameCallback) {
      videoRef.current.cancelVideoFrameCallback(frameCallbackRef.current)
    }
    frameCallbackRef.current = null
    const stream = videoRef.current?.srcObject
    stream?.getTracks?.().forEach(t => t.stop())
    if (videoRef.current) videoRef.current.srcObject = null
    socketRef.current?.close()
    socketRef.current = null
    setCameraOn(false)
    setConnected(false)
    setVitals({ status: 'idle', hr: null, rr: null, hrv_rmssd: null, signal_quality: 0, waveform: [], bbox: null, window_seconds: 0, measurement_progress: 0 })
  }

  const finishCapture = () => {
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = null
    captureActiveRef.current = false
    if (frameCallbackRef.current != null && videoRef.current?.cancelVideoFrameCallback) {
      videoRef.current.cancelVideoFrameCallback(frameCallbackRef.current)
    }
    frameCallbackRef.current = null
    videoRef.current?.srcObject?.getTracks?.().forEach(track => track.stop())
    if (videoRef.current) videoRef.current.srcObject = null
    socketRef.current?.close()
    socketRef.current = null
    setCameraOn(false)
    setConnected(false)
  }

  const returnToMeasurement = () => {
    savedResultRef.current = false
    setVitals({ status: 'idle', hr: null, rr: null, hrv_rmssd: null, signal_quality: 0, waveform: [], bbox: null, window_seconds: 0, measurement_progress: 0 })
  }

  const deleteHistory = id => setHistory(current => {
    const next = current.filter(record => record.id !== id)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
    return next
  })
  const clearHistory = () => { localStorage.removeItem(HISTORY_KEY); setHistory([]) }

  const sendFrame = (captureTimestamp) => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const ws = socketRef.current
    if (!video || !canvas || video.readyState < 2) return
    if (!useLocalAnalysis && (ws?.readyState !== WebSocket.OPEN || ws.bufferedAmount > 750_000)) return
    const w = 640
    const h = Math.round(w * (video.videoHeight || 720) / (video.videoWidth || 1280))
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d', { alpha: false })
    ctx.drawImage(video, 0, 0, w, h)
    if (useLocalAnalysis) {
      localAnalyzerRef.current?.process(ctx, w, h, captureTimestamp, engine)
      return
    }
    // High-quality JPEG reduces chroma quantization while staying fast enough for WebSocket transport.
    const image = canvas.toDataURL('image/jpeg', 0.94)
    ws.send(JSON.stringify({ type: 'frame', timestamp: captureTimestamp, image }))
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
  const progress = Math.min(100, Math.round(vitals.measurement_progress ?? (vitals.window_seconds || 0) * 2))

  if (showHistory) {
    return <HistoryView records={history} onBack={() => setShowHistory(false)} onDelete={deleteHistory} onClear={clearHistory} />
  }

  if (vitals.status === 'complete') {
    return <ResultsView vitals={vitals} onBack={returnToMeasurement} onHistory={() => setShowHistory(true)} />
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="hero-copy">
          <div className="eyebrow"><Radio size={15} /> rPPG 即時量測系統</div>
          <h1>讓鏡頭讀懂<br/><span>你的生命訊號。</span></h1>
          <p>以一般 RGB 鏡頭擷取臉部微小色彩變化，非接觸量測心率、心率變異、呼吸與實驗性健康指標。</p>
        </div>
        <div className="status-stack">
          <div className={`connection ${connected ? 'online' : ''}`}>
          <span className="dot" /> {(connected || (useLocalAnalysis && cameraOn)) ? '系統量測中' : '系統待機中'}
          </div>
          <div className="trust-chips"><span>50 秒</span><span>非接觸式</span><span>裝置內分析</span></div>
          <button className="history-button" onClick={() => setShowHistory(true)}><History size={16} /> 歷史紀錄 {history.length > 0 && <b>{history.length}</b>}</button>
        </div>
      </header>

      <section className="layout">
        <div className="camera-panel panel">
          <div className="panel-head">
            <div><Camera size={18} /> 臉部訊號擷取 <small>01／影像擷取</small></div>
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
              }}><span>臉部取樣區</span></div>
            )}
            {cameraOn && <div className="scan-line" />}
          </div>
          <canvas ref={canvasRef} className="capture-canvas" />

          <div className="signal-strip capture-metrics">
            <MetricCard label="訊號品質 SQI" value={cameraOn ? qPct : null} unit="%" hint="含頻譜、動作、曝光" />
            <MetricCard label="有效量測" value={Math.round(vitals.window_seconds || 0)} unit="秒" hint="目標 50 個有效秒數" />
          </div>

          <div className="camera-controls">
            <button className="primary" onClick={cameraOn ? stopCamera : startCamera}>{cameraOn ? '停止量測' : '啟動鏡頭'}</button>
            <div className="capture-note"><strong>準備好了嗎？</strong><span>完整量測約 50 秒；距離 50–80 cm、光線固定並保持不動</span></div>
          </div>

          {cameraOn && (
            <div className="quality-box">
              <div className="quality-row">
                <strong className={qClass}>{qLabel}</strong>
                <span>{vitals.status === 'complete' ? '100% 完成' : `${Math.round(vitals.window_seconds || 0)} / 50 有效秒數`}</span>
              </div>
              <div className="progress"><span style={{ width: `${progress}%` }} /></div>
              <div className="quality-tips">
                <span>動作 {vitals.motion == null ? '—' : Math.round(vitals.motion * 100) + '%'}</span>
                <span>曝光 {vitals.exposure == null ? '—' : Math.round(vitals.exposure * 100) + '%'}</span>
                <span>取樣區 {vitals.roi_quality == null ? '—' : Math.round(vitals.roi_quality * 100) + '%'}</span>
                <span>影格率 {vitals.fps ?? '—'}</span>
              </div>
            </div>
          )}
          {error && <div className="error-box">{error}</div>}
        </div>

        <div className="right-column">
          <VitalsBody vitals={vitals} live={cameraOn} />

          {vitals.wellness && <WellnessRadar values={vitals.wellness} />}

          <div className="panel disclaimer">
            <ShieldCheck size={20} />
            <div><strong>輔助感測，不是醫療診斷</strong><p>本系統以一般 RGB 網路攝影機估算生理訊號；動作、照明、膚色、影像壓縮及鏡頭自動曝光等都可能影響結果。請勿作為疾病診斷、用藥或緊急醫療判斷依據。</p></div>
          </div>
        </div>
      </section>
    </main>
  )
}
