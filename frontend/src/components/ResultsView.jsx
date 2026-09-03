import { ArrowLeft, HeartPulse, History } from 'lucide-react'
import WellnessRadar from './WellnessRadar'
import VitalsBody from './VitalsBody'

export default function ResultsView({ vitals, onBack, onHistory }) {
  return (
    <main className="results-shell">
      <header className="results-header">
        <button className="back-button" onClick={onBack}><ArrowLeft size={20} /> 重新量測</button>
        <div className="results-brand"><HeartPulse size={22} /> rPPG <span>生理數據報告</span></div>
        <div className="results-actions"><button className="history-button" onClick={onHistory}><History size={16} /> 歷史紀錄</button><span className="complete-badge">量測完成</span></div>
      </header>

      <section className="results-content">
        <VitalsBody vitals={vitals} />
        <div className="results-side">
          <WellnessRadar values={vitals.wellness} showEmpty />
          <div className="result-summary panel">
            <h2>本次量測摘要</h2>
            <div><span>有效量測時間</span><strong>{Math.round(vitals.window_seconds || 0)} 秒</strong></div>
            <div><span>訊號品質</span><strong>{Math.round((vitals.signal_quality || 0) * 100)}%</strong></div>
            <div><span>分析引擎</span><strong>{vitals.engine?.toUpperCase?.() || '—'}</strong></div>
            <p>血壓與血氧皆為未校正的 RGB 代理估算，不能取代袖帶式血壓計或指夾式血氧計。負荷趨勢與六維圖為本系統的實驗性 HRV 衍生結果。</p>
          </div>
        </div>
      </section>
    </main>
  )
}
