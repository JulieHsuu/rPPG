export default function Waveform({ values = [] }) {
  const width = 700
  const height = 150
  if (!values.length) {
    return <div className="wave-empty">累積影像訊號中…</div>
  }
  const points = values.map((v, i) => {
    const x = (i / Math.max(values.length - 1, 1)) * width
    const y = height / 2 - v * (height * 0.38)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  return (
    <svg className="wave" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label="rPPG waveform">
      <line x1="0" y1={height / 2} x2={width} y2={height / 2} className="wave-axis" />
      <polyline points={points} className="wave-line" />
    </svg>
  )
}
