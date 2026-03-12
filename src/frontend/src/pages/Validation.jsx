import { useMemo, useState } from 'react'
import { ResponsiveContainer, ComposedChart, Area, Line, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ScatterChart, ReferenceLine, BarChart, Bar } from 'recharts'
import { validateMetrics } from '../api/client'

export default function Validation() {
  const [vpcForm, setVpcForm] = useState({ ID: '', TIME: '', DV: '', PRED: '', DOSE: '', WT: '' })
  const [vpcQueue, setVpcQueue] = useState([])
  const [baForm, setBaForm] = useState({ Patient: '', Method1: '', Method2: '' })
  const [baQueue, setBaQueue] = useState([])
  const [npdeForm, setNpdeForm] = useState({ ID: '', TIME: '', DV: '', PRED: '', IPRED: '' })
  const [npdeQueue, setNpdeQueue] = useState([])
  const [vpcRows, setVpcRows] = useState([])
  const [baRows, setBaRows] = useState([])
  const [npdeRows, setNpdeRows] = useState([])
  const [apiMetrics, setApiMetrics] = useState(null)
  const [apiLoading, setApiLoading] = useState(false)
  const [apiError, setApiError] = useState(null)

  function quantile(arr, q) {
    if (!arr.length) return 0
    const s = [...arr].sort((a, b) => a - b)
    const pos = (s.length - 1) * q
    const base = Math.floor(pos)
    const rest = pos - base
    if (s[base + 1] !== undefined) return s[base] + rest * (s[base + 1] - s[base])
    return s[base]
  }

  function groupByTime(rows) {
    const m = new Map()
    rows.forEach(r => {
      const t = +r.TIME
      if (!m.has(t)) m.set(t, [])
      m.get(t).push(r)
    })
    const times = Array.from(m.keys()).sort((a, b) => a - b)
    return times.map(t => {
      const rs = m.get(t)
      const preds = rs.map(r => +r.PRED)
      return { t, lo: quantile(preds, 0.05), hi: quantile(preds, 0.95), med: quantile(preds, 0.5) }
    })
  }

  const vpcBands = useMemo(() => groupByTime(vpcRows), [vpcRows])
  const vpcDvPoints = useMemo(() => vpcRows.map(r => ({ t: +r.TIME, dv: +r.DV })), [vpcRows])

  function parseVpc() {
    setVpcRows(vpcQueue)
  }
  function parseBa() {
    setBaRows(baQueue.map(r => ({ Patient: r.Patient, Method1: +r.Method1, Method2: +r.Method2 })))
  }
  function parseNpde() {
    setNpdeRows(npdeQueue.map(r => ({ ID: r.ID, TIME: +r.TIME, DV: +r.DV, PRED: +r.PRED, IPRED: +r.IPRED })))
  }

  const baData = useMemo(() => {
    const pts = baRows.map(r => ({ mean: (r.Method1 + r.Method2) / 2, diff: r.Method1 - r.Method2 }))
    const meanDiff = pts.length ? pts.reduce((a, b) => a + b.diff, 0) / pts.length : 0
    const sd = pts.length ? Math.sqrt(pts.reduce((a, b) => a + Math.pow(b.diff - meanDiff, 2), 0) / pts.length) : 0
    return { pts, meanDiff, sd }
  }, [baRows])

  const npdeData = useMemo(() => {
    const resids = npdeRows.map(r => r.DV - r.IPRED)
    const mean = resids.length ? resids.reduce((a, b) => a + b, 0) / resids.length : 0
    const sd = resids.length ? Math.sqrt(resids.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / resids.length) : 1
    const z = resids.map(u => (u - mean) / (sd || 1))
    const histBins = 12
    const minZ = Math.min(...z, -3)
    const maxZ = Math.max(...z, 3)
    const step = (maxZ - minZ) / histBins || 1
    const hist = Array.from({ length: histBins }).map((_, i) => ({ bin: +(minZ + i * step).toFixed(2), count: 0 }))
    z.forEach(v => {
      const idx = Math.min(histBins - 1, Math.max(0, Math.floor((v - minZ) / step)))
      hist[idx].count += 1
    })
    const sortedZ = [...z].sort((a, b) => a - b)
    const qq = sortedZ.map((v, i) => {
      const p = (i + 0.5) / sortedZ.length
      const theo = normInv(p)
      return { theo, sample: v }
    })
    return { hist, qq, mean, sd }
  }, [npdeRows])

  function normInv(p) {
    const a1 = -39.69683028665376, a2 = 220.9460984245205, a3 = -275.9285104469687
    const a4 = 138.3577518672690, a5 = -30.66479806614716, a6 = 2.506628277459239
    const b1 = -54.47609879822406, b2 = 161.5858368580409, b3 = -155.6989798598866
    const b4 = 66.80131188771972, b5 = -13.28068155288572
    const c1 = -0.007784894002430293, c2 = -0.3223964580411365, c3 = -2.400758277161838
    const c4 = -2.549732539343734, c5 = 4.374664141464968, c6 = 2.938163982698783
    const d1 = 0.007784695709041462, d2 = 0.3224671290700398, d3 = 2.445134137142996, d4 = 3.754408661907416
    const plow = 0.02425, phigh = 1 - plow
    let q, r
    if (p < plow) {
      q = Math.sqrt(-2 * Math.log(p))
      return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    }
    if (phigh < p) {
      q = Math.sqrt(-2 * Math.log(1 - p))
      return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    }
    q = p - 0.5
    r = q * q
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
      (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
  }

  return (
    <div className="grid">
      {/* API Validation Panel */}
      <div className="card">
        <h2>🧪 Thẩm định bằng API (Công việc 4.2)</h2>
        <p style={{fontSize: '0.85em', color: '#666'}}>
          Gửi dữ liệu xuống backend để tính MPE, MAPE, CCC, NPDE, Coverage.
          Dữ liệu từ bảng VPC và Bland-Altman bên dưới sẽ được sử dụng.
        </p>
        <button
          className="btn primary"
          disabled={apiLoading}
          onClick={async () => {
            setApiLoading(true)
            setApiError(null)
            try {
              const estimated = vpcRows.length > 0
                ? vpcRows.map(r => +r.PRED)
                : baQueue.map(r => +r.Method1)
              const trueValues = vpcRows.length > 0
                ? vpcRows.map(r => +r.DV)
                : baQueue.map(r => +r.Method2)
              if (estimated.length === 0) {
                setApiError('Chưa có dữ liệu. Hãy nhập VPC hoặc Bland-Altman trước.')
                return
              }
              const res = await validateMetrics({ estimated, true_values: trueValues })
              setApiMetrics(res)
            } catch (err) {
              setApiError(err.message)
            } finally {
              setApiLoading(false)
            }
          }}
        >
          {apiLoading ? 'Đang tính...' : 'Chạy Validation API'}
        </button>
        {apiError && <p style={{color:'#e53e3e', marginTop:8}}>⚠ {apiError}</p>}
        {apiMetrics && (
          <div style={{display:'grid', gridTemplateColumns:'repeat(5, 1fr)', gap:12, marginTop:12}}>
            {[
              { label: 'MPE (%)', value: apiMetrics.mpe?.toFixed(2), ok: Math.abs(apiMetrics.mpe||0) < 15 },
              { label: 'MAPE (%)', value: apiMetrics.mape?.toFixed(2), ok: (apiMetrics.mape||100) < 20 },
              { label: 'CCC', value: apiMetrics.ccc?.toFixed(3), ok: (apiMetrics.ccc||0) > 0.85 },
              { label: 'NPDE p-value', value: apiMetrics.npde_pvalue?.toFixed(3), ok: (apiMetrics.npde_pvalue||0) > 0.05 },
              { label: 'Coverage 95%', value: apiMetrics.coverage_95 != null ? `${(apiMetrics.coverage_95*100).toFixed(1)}%` : '—', ok: (apiMetrics.coverage_95||0) > 0.85 },
            ].map(m => (
              <div key={m.label} style={{padding:'12px', background: m.ok ? '#f0fdf4' : '#fef2f2', borderRadius:8, textAlign:'center'}}>
                <div style={{fontSize:11, color:'#666'}}>{m.label}</div>
                <div style={{fontSize:22, fontWeight:800, color: m.ok ? '#16a34a' : '#dc2626'}}>{m.value || '—'}</div>
                <div style={{fontSize:10}}>{m.ok ? '✅ Đạt' : '⚠ Cần xem lại'}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid">
        <div className="card">
          <h2>VPC input</h2>
          <div className="form" style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
            <label>ID<input value={vpcForm.ID} onChange={e => setVpcForm(f => ({ ...f, ID: e.target.value }))} /></label>
            <label>TIME<input type="number" value={vpcForm.TIME} onChange={e => setVpcForm(f => ({ ...f, TIME: e.target.value }))} /></label>
            <label>DV<input type="number" value={vpcForm.DV} onChange={e => setVpcForm(f => ({ ...f, DV: e.target.value }))} /></label>
            <label>PRED<input type="number" value={vpcForm.PRED} onChange={e => setVpcForm(f => ({ ...f, PRED: e.target.value }))} /></label>
            <label>DOSE<input type="number" value={vpcForm.DOSE} onChange={e => setVpcForm(f => ({ ...f, DOSE: e.target.value }))} /></label>
            <label>WT<input type="number" value={vpcForm.WT} onChange={e => setVpcForm(f => ({ ...f, WT: e.target.value }))} /></label>
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button className="btn" onClick={() => { setVpcQueue(q => [...q, { ...vpcForm }]); setVpcForm({ ID: '', TIME: '', DV: '', PRED: '', DOSE: '', WT: '' }) }}>Thêm dòng</button>
            <button className="btn" onClick={() => setVpcQueue([])}>Xoá tất cả</button>
            <button className="btn primary" onClick={parseVpc}>Parse</button>
          </div>
          <div style={{ overflowX: 'auto', marginTop: 8 }}>
            <table className="table">
              <thead><tr><th>ID</th><th>TIME</th><th>DV</th><th>PRED</th><th>DOSE</th><th>WT</th><th></th></tr></thead>
              <tbody>
                {vpcQueue.map((r, i) => (
                  <tr key={i}>
                    <td>{r.ID}</td><td>{r.TIME}</td><td>{r.DV}</td><td>{r.PRED}</td><td>{r.DOSE}</td><td>{r.WT}</td>
                    <td><button className="btn" onClick={() => setVpcQueue(q => q.filter((_, j) => j !== i))}>Xoá</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <h3>VPC chart</h3>
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={vpcBands}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="hi" stroke="#cbd5e1" fill="#e2e8f0" />
              <Area type="monotone" dataKey="lo" stroke="#cbd5e1" fill="#e2e8f0" />
              <Line type="monotone" dataKey="med" stroke="#0ea5e9" strokeWidth={2} dot={false} />
              <Scatter data={vpcDvPoints} fill="#ef4444" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h2>Bland–Altman input</h2>
          <div className="form" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
            <label>Patient<input value={baForm.Patient} onChange={e => setBaForm(f => ({ ...f, Patient: e.target.value }))} /></label>
            <label>Method1<input type="number" value={baForm.Method1} onChange={e => setBaForm(f => ({ ...f, Method1: e.target.value }))} /></label>
            <label>Method2<input type="number" value={baForm.Method2} onChange={e => setBaForm(f => ({ ...f, Method2: e.target.value }))} /></label>
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button className="btn" onClick={() => { setBaQueue(q => [...q, { ...baForm }]); setBaForm({ Patient: '', Method1: '', Method2: '' }) }}>Thêm dòng</button>
            <button className="btn" onClick={() => setBaQueue([])}>Xoá tất cả</button>
            <button className="btn primary" onClick={parseBa}>Parse</button>
          </div>
          <div style={{ overflowX: 'auto', marginTop: 8 }}>
            <table className="table">
              <thead><tr><th>Patient</th><th>Method1</th><th>Method2</th><th></th></tr></thead>
              <tbody>
                {baQueue.map((r, i) => (
                  <tr key={i}>
                    <td>{r.Patient}</td><td>{r.Method1}</td><td>{r.Method2}</td>
                    <td><button className="btn" onClick={() => setBaQueue(q => q.filter((_, j) => j !== i))}>Xoá</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <h3>Bland–Altman plot</h3>
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="mean" />
              <YAxis dataKey="diff" />
              <Tooltip />
              <Scatter data={baData.pts} fill="#0ea5e9" />
              <ReferenceLine y={baData.meanDiff} stroke="#10b981" />
              <ReferenceLine y={baData.meanDiff + 1.96 * baData.sd} stroke="#ef4444" strokeDasharray="4 4" />
              <ReferenceLine y={baData.meanDiff - 1.96 * baData.sd} stroke="#ef4444" strokeDasharray="4 4" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h2>NPDE input</h2>
          <div className="form" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
            <label>ID<input value={npdeForm.ID} onChange={e => setNpdeForm(f => ({ ...f, ID: e.target.value }))} /></label>
            <label>TIME<input type="number" value={npdeForm.TIME} onChange={e => setNpdeForm(f => ({ ...f, TIME: e.target.value }))} /></label>
            <label>DV<input type="number" value={npdeForm.DV} onChange={e => setNpdeForm(f => ({ ...f, DV: e.target.value }))} /></label>
            <label>PRED<input type="number" value={npdeForm.PRED} onChange={e => setNpdeForm(f => ({ ...f, PRED: e.target.value }))} /></label>
            <label>IPRED<input type="number" value={npdeForm.IPRED} onChange={e => setNpdeForm(f => ({ ...f, IPRED: e.target.value }))} /></label>
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button className="btn" onClick={() => { setNpdeQueue(q => [...q, { ...npdeForm }]); setNpdeForm({ ID: '', TIME: '', DV: '', PRED: '', IPRED: '' }) }}>Thêm dòng</button>
            <button className="btn" onClick={() => setNpdeQueue([])}>Xoá tất cả</button>
            <button className="btn primary" onClick={parseNpde}>Parse</button>
          </div>
          <div style={{ overflowX: 'auto', marginTop: 8 }}>
            <table className="table">
              <thead><tr><th>ID</th><th>TIME</th><th>DV</th><th>PRED</th><th>IPRED</th><th></th></tr></thead>
              <tbody>
                {npdeQueue.map((r, i) => (
                  <tr key={i}>
                    <td>{r.ID}</td><td>{r.TIME}</td><td>{r.DV}</td><td>{r.PRED}</td><td>{r.IPRED}</td>
                    <td><button className="btn" onClick={() => setNpdeQueue(q => q.filter((_, j) => j !== i))}>Xoá</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <h3>NPDE histogram</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={npdeData.hist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
          <h3>NPDE QQ plot</h3>
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="theo" />
              <YAxis dataKey="sample" />
              <Tooltip />
              <Scatter data={npdeData.qq} fill="#14b8a6" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
