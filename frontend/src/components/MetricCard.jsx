export default function MetricCard({ label, value, unit, hint, accent = false }) {
  return (
    <div className={`metric-card ${accent ? 'accent' : ''}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value-row">
        <span className="metric-value">{value ?? '—'}</span>
        <span className="metric-unit">{unit}</span>
      </div>
      <div className="metric-hint">{hint}</div>
    </div>
  )
}
