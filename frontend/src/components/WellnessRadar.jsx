const AXES = [
  ['activity', '活動'],
  ['sleep', '睡眠'],
  ['equilibrium', '平衡'],
  ['metabolism', '代謝'],
  ['health', '健康'],
  ['relaxation', '放鬆'],
]

function point(index, value, radius = 118) {
  const angle = -Math.PI / 2 + index * Math.PI / 3
  const distance = radius * value / 5
  return [210 + Math.cos(angle) * distance, 165 + Math.sin(angle) * distance]
}

function polygon(value) {
  return AXES.map((_, index) => point(index, value).join(',')).join(' ')
}

export default function WellnessRadar({ values, showEmpty = false }) {
  if (!values && !showEmpty) return null
  const dataPoints = values ? AXES.map(([key], index) => point(index, values[key] || 1).join(',')).join(' ') : null

  return (
    <div className="wellness-panel panel">
      <div className="panel-head">
        <div>HRV 衍生六維指標</div>
        <span className="mono">實驗性 · 1–5 分</span>
      </div>
      <svg className="radar" viewBox="0 0 420 345" role="img" aria-label="六維實驗性 wellness 雷達圖">
        {[1, 2, 3, 4, 5].map(level => (
          <polygon key={level} points={polygon(level)} className="radar-grid" />
        ))}
        {AXES.map(([, label], index) => {
          const [x, y] = point(index, 5)
          const [lx, ly] = point(index, 6.05)
          return (
            <g key={label}>
              <line x1="210" y1="165" x2={x} y2={y} className="radar-axis" />
              <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" className="radar-label">{label}</text>
            </g>
          )
        })}
        {dataPoints && <polygon points={dataPoints} className="radar-data" />}
        {values && AXES.map(([key, label], index) => {
          const [x, y] = point(index, values[key] || 1)
          return <circle key={label} cx={x} cy={y} r="4" className="radar-point" />
        })}
      </svg>
      {values ? <div className="wellness-values">
        {AXES.map(([key, label]) => <span key={key}>{label}<strong>{values[key]?.toFixed?.(1) ?? '—'}</strong></span>)}
      </div> : <div className="radar-empty">本次 HRV 訊號不足，無法產生六維結果</div>}
      <p className="proxy-note">依本次心率、RMSSD、動作與訊號品質計算的介面示範，不等同 FaceHeart 專有模型，也不是睡眠、代謝或健康診斷。</p>
    </div>
  )
}
