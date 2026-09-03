import { Activity, Brain, HeartPulse, Wind } from 'lucide-react'

function fallbackWave(kind, count = 150) {
  return Array.from({ length: count }, (_, i) => {
    const p = (i % 42) / 42
    if (kind === 'ecg') {
      if (p < .12) return .06 * Math.sin(p / .12 * Math.PI)
      if (p < .25) return 0
      if (p < .29) return -.18
      if (p < .32) return 1
      if (p < .36) return -.35
      if (p < .52) return 0
      if (p < .75) return .14 * Math.sin((p - .52) / .23 * Math.PI)
      return 0
    }
    if (kind === 'spo2') return Math.pow(Math.max(0, Math.sin(p * Math.PI)), 2.4) * .72 - .18
    return Math.sin(i / count * Math.PI * 4) * .52
  })
}

function MonitorTrace({ label, unit, color, values, kind, value }) {
  const source = values?.length ? values : fallbackWave(kind)
  const width = 700; const height = 118
  const points = source.map((v, i) => `${(i / Math.max(1, source.length - 1) * width).toFixed(1)},${(height / 2 - v * height * .34).toFixed(1)}`).join(' ')
  return <div className="monitor-trace" style={{ '--trace': color }}>
    <div className="trace-label"><span>{label}</span><strong>{value ?? '—'} <small>{value != null ? unit : ''}</small></strong></div>
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label={`${label} 波形`}><polyline points={points} /></svg>
    <span className="trace-scan" />
  </div>
}

function Reading({ icon, label, value, unit, tone, note }) {
  return <div className={`monitor-reading ${tone}`}><div className="reading-icon">{icon}</div><div className="reading-copy"><span>{label}</span><strong>{value ?? '—'} <small>{value != null ? unit : ''}</small></strong>{note && <em>{note}</em>}</div></div>
}

export default function VitalsBody({ vitals, live = false }) {
  const stress = vitals.stress_index == null ? null : vitals.stress_index < 60 ? '一般' : vitals.stress_index < 75 ? '偏高' : '高'
  const smoothWave = vitals.waveform?.map((_, i, a) => (a[Math.max(0, i - 2)] + a[Math.max(0, i - 1)] + a[i]) / 3)
  const rrWave = Array.from({ length: 150 }, (_, i) => Math.sin(i / 150 * Math.PI * 4))
  return <section className={`patient-monitor panel ${live ? 'is-live' : 'is-result'}`}>
    <div className="monitor-head"><div><span className="live-dot" /> 生理訊號監視器 <small>02／訊號分析</small></div><span>{live ? '即時監測中' : '本次量測結果'}</span></div>
    <div className="monitor-screen">
      <div className="monitor-traces">
        <MonitorTrace label="心搏波形／rPPG" value={vitals.hr} unit="次／分" color="#5ce2bd" values={vitals.waveform} kind="ecg" />
        <MonitorTrace label="血氧脈波" value={vitals.spo2} unit="SpO₂%" color="#68c9ff" values={smoothWave} kind="spo2" />
        <MonitorTrace label="呼吸波形" value={vitals.rr} unit="次／分" color="#ffd166" values={vitals.rr ? rrWave : null} kind="resp" />
      </div>
      <aside className="monitor-readings">
        <Reading tone="heart" icon={<HeartPulse />} label="心率" value={vitals.hr} unit="次／分" />
        <Reading tone="hrv" icon={<Activity />} label="心率變異 HRV" value={vitals.hrv_rmssd} unit="毫秒" />
        <Reading tone="pressure" icon={<Activity />} label="血壓推估" value={vitals.bp_systolic != null ? `${vitals.bp_systolic}/${vitals.bp_diastolic}` : null} unit="mmHg" note="未校正，非血壓計" />
        <Reading tone="oxygen" icon={<Activity />} label="血氧濃度 SpO₂" value={vitals.spo2} unit="%" note="RGB 實驗估算" />
        <Reading tone="resp" icon={<Wind />} label="呼吸率" value={vitals.rr} unit="次／分" />
        <Reading tone="stress" icon={<Brain />} label="負荷趨勢" value={stress} unit="" note="實驗性指標" />
      </aside>
    </div>
    <div className="monitor-foot"><span>RGB 光學訊號</span><span>{vitals.fps ? `每秒 ${vitals.fps} 影格` : '等待訊號中'}</span><span>不可用於醫療診斷</span></div>
  </section>
}
