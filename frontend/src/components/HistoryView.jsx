import { ArrowLeft, Download, History, Trash2 } from 'lucide-react'

const value = (v, suffix = '') => v == null ? '—' : `${v}${suffix}`
const csvCell = v => `"${String(v ?? '').replaceAll('"', '""')}"`

function exportCsv(records) {
  const columns = [
    ['量測編號', 'id'], ['量測時間', 'measured_at'], ['分析引擎', 'engine'], ['訊號品質(%)', 'sqi'], ['有效量測(秒)', 'seconds'],
    ['本站心率(bpm)', 'hr'], ['FaceHeart心率(bpm)', 'fh_hr'], ['本站HRV(ms)', 'hrv'], ['FaceHeart HRV(ms)', 'fh_hrv'],
    ['本站血氧(%)', 'spo2'], ['FaceHeart血氧(%)', 'fh_spo2'], ['本站呼吸(次/分)', 'rr'], ['FaceHeart呼吸(次/分)', 'fh_rr'],
    ['本站收縮壓(mmHg)', 'sbp'], ['本站舒張壓(mmHg)', 'dbp'], ['FaceHeart收縮壓(mmHg)', 'fh_sbp'], ['FaceHeart舒張壓(mmHg)', 'fh_dbp'],
    ['本站負荷指數', 'stress'], ['FaceHeart壓力', 'fh_stress'], ['活動', 'activity'], ['睡眠', 'sleep'], ['平衡', 'equilibrium'],
    ['代謝', 'metabolism'], ['健康', 'health'], ['放鬆', 'relaxation'], ['備註', 'notes'],
  ]
  const rows = [columns.map(([label]) => csvCell(label)).join(',')]
  for (const record of records) rows.push(columns.map(([, key]) => csvCell(record[key])).join(','))
  const blob = new Blob([`\ufeff${rows.join('\r\n')}`], { type: 'text/csv;charset=utf-8' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `rPPG量測紀錄_${new Date().toISOString().slice(0, 10)}.csv`
  link.click()
  URL.revokeObjectURL(link.href)
}

export default function HistoryView({ records, onBack, onDelete, onClear }) {
  return <main className="history-shell">
    <header className="history-header">
      <button className="back-button" onClick={onBack}><ArrowLeft size={19} /> 返回量測</button>
      <div className="history-title"><History size={21} /><div><strong>量測歷史紀錄</strong><span>資料只儲存在目前瀏覽器</span></div></div>
      <button className="export-button" disabled={!records.length} onClick={() => exportCsv(records)}><Download size={17} /> 匯出 Excel CSV</button>
    </header>

    <section className="history-panel panel">
      <div className="history-toolbar"><div><strong>{records.length}</strong> 筆量測資料</div>{records.length > 0 && <button className="clear-button" onClick={() => window.confirm('確定刪除全部歷史紀錄？此操作無法復原。') && onClear()}><Trash2 size={15} /> 全部刪除</button>}</div>
      {!records.length ? <div className="history-empty"><History size={42} /><strong>尚無量測紀錄</strong><span>完成第一次 50 秒量測後，結果會自動保存在這裡。</span></div> :
        <div className="history-table-wrap"><table className="history-table">
          <thead><tr><th>量測時間</th><th>心率</th><th>HRV</th><th>血壓推估</th><th>血氧</th><th>呼吸</th><th>負荷</th><th>SQI</th><th>引擎</th><th aria-label="操作" /></tr></thead>
          <tbody>{records.map(record => <tr key={record.id}>
            <td><strong>{new Date(record.measured_at).toLocaleDateString('zh-TW')}</strong><span>{new Date(record.measured_at).toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' })}</span></td>
            <td>{value(record.hr)}</td><td>{value(record.hrv)}</td><td>{record.sbp == null ? '—' : `${record.sbp}/${record.dbp}`}</td><td>{value(record.spo2, '%')}</td><td>{value(record.rr)}</td><td>{value(record.stress)}</td><td>{value(record.sqi, '%')}</td><td>{record.engine || '—'}</td>
            <td><button className="delete-record" title="刪除此筆" aria-label={`刪除 ${new Date(record.measured_at).toLocaleString('zh-TW')} 的紀錄`} onClick={() => window.confirm('確定刪除這筆量測紀錄？') && onDelete(record.id)}><Trash2 size={16} /></button></td>
          </tr>)}</tbody>
        </table></div>}
      <p className="history-note">CSV 已預留 FaceHeart 對照欄位。使用 Excel 開啟後填入同次量測數值，再將檔案交給我即可進行配對校正。</p>
    </section>
  </main>
}
