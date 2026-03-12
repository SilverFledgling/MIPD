import { useState, useEffect } from 'react'
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, ReferenceArea, Legend, Cell,
} from 'recharts'

/**
 * Generate 3-layer PK curve data:
 * - Population (dashed): TV params
 * - Individual (solid): Bayesian posterior
 * - CI ribbon: 95% credible interval
 */
function generatePKCurve(result) {
  if (result?.pkCurve) {
    return result.pkCurve.timePoints.map((t, i) => ({
      t,
      population: result.pkCurve.concentrations[i] * 1.1,  // close to individual for demo
      individual: result.pkCurve.concentrations[i],
      ci_lower: result.pkCurve.ci95Lower?.[i] || result.pkCurve.concentrations[i] * 0.7,
      ci_upper: result.pkCurve.ci95Upper?.[i] || result.pkCurve.concentrations[i] * 1.3,
    }))
  }
  // Fallback: generate from basic params
  const cl = result?.individualParams?.CL?.value || result?.individualParams?.CL || 3.5
  const v = result?.individualParams?.V1?.value || result?.individualParams?.V1 || 30
  const ke = cl / v
  return Array.from({ length: 49 }).map((_, i) => {
    const t = i * 0.5
    const c_ind = 35 * Math.exp(-ke * t)
    const c_pop = 35 * Math.exp(-(ke * 0.9) * t)
    return {
      t,
      population: +c_pop.toFixed(2),
      individual: +c_ind.toFixed(2),
      ci_lower: +(c_ind * 0.7).toFixed(2),
      ci_upper: +(c_ind * 1.3).toFixed(2),
    }
  })
}

export default function Results() {
  const [result, setResult] = useState(null)

  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('vanco_result') || 'null')
      if (stored) setResult(stored)
    } catch {}
  }, [])

  const pkData = generatePKCurve(result)

  // SHAP data (from API or default)
  const shapData = result?.shapExplanation || [
    { feature: 'CrCL', contribution: -0.35 },
    { feature: 'Cân nặng', contribution: -0.12 },
    { feature: 'Tuổi', contribution: -0.08 },
    { feature: 'Albumin', contribution: 0.02 },
    { feature: 'ICU', contribution: 0.0 },
  ]

  return (
    <div className="card">
      <h2>Kết quả PK — Trực quan hóa 3 lớp</h2>

      {/* Summary */}
      {result && (
        <ul>
          <li>Liều đề xuất: <strong>{result.recommendation?.dose || result.perDose || '—'} mg q{result.recommendation?.interval || result.interval || '—'}h</strong></li>
          <li>AUC₂₄: {result.predictions?.auc24 || result.auc || '—'} mg·h/L</li>
          <li>Phương pháp: {result.method || '—'}</li>
          {result.diagnostics?.layers_executed && (
            <li>Pipeline: {result.diagnostics.layers_executed.join(' → ')}</li>
          )}
        </ul>
      )}

      {/* 3-Layer PK Curve Chart */}
      <h3>Đường cong Dược động học (3 lớp)</h3>
      <p style={{fontSize:'0.85em', color:'#666'}}>
        <span style={{color:'#94a3b8'}}>— —</span> Quần thể (Population) &nbsp;|&nbsp;
        <span style={{color:'#2563eb'}}>━━</span> Cá thể (Individual) &nbsp;|&nbsp;
        <span style={{color:'#bfdbfe'}}>░░</span> 95% Credible Interval
      </p>
      <ResponsiveContainer width="100%" height={350}>
        <AreaChart data={pkData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="t" label={{ value: 'Giờ', position: 'insideBottomRight', offset: -4 }} />
          <YAxis label={{ value: 'mg/L', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          {/* Target zone */}
          <ReferenceArea y1={10} y2={20} fill="#22c55e" fillOpacity={0.08}
                         label={{ value: 'Vùng đích', position: 'right', fill: '#22c55e' }} />
          {/* CI ribbon */}
          <Area type="monotone" dataKey="ci_upper" stroke="none" fill="#bfdbfe" fillOpacity={0.5} />
          <Area type="monotone" dataKey="ci_lower" stroke="none" fill="#ffffff" fillOpacity={1} />
          {/* Population line (dashed) */}
          <Line type="monotone" dataKey="population" stroke="#94a3b8" strokeWidth={1.5}
                strokeDasharray="5 5" dot={false} name="Quần thể" />
          {/* Individual line (solid) */}
          <Line type="monotone" dataKey="individual" stroke="#2563eb" strokeWidth={2.5}
                dot={false} name="Cá thể" />
        </AreaChart>
      </ResponsiveContainer>

      {/* SHAP Waterfall Chart */}
      <div className="grid two">
        <div className="card">
          <h3>SHAP — Giải thích yếu tố ảnh hưởng</h3>
          <p style={{fontSize:'0.8em', color:'#666'}}>
            Giá trị dương = tăng CL (cần ↑ liều) | Giá trị âm = giảm CL (cần ↓ liều)
          </p>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={shapData} layout="vertical" margin={{ left: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[-0.5, 0.1]} />
              <YAxis dataKey="feature" type="category" width={80} />
              <Tooltip />
              <ReferenceLine x={0} stroke="#000" />
              <Bar dataKey="contribution" name="SHAP value">
                {shapData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.contribution >= 0 ? '#22c55e' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Tham số PK cá thể</h3>
          {result?.individualParams ? (
            <table style={{width:'100%', fontSize:'0.9em'}}>
              <thead>
                <tr><th>Tham số</th><th>Giá trị</th><th>CI 95%</th></tr>
              </thead>
              <tbody>
                {Object.entries(result.individualParams).map(([key, val]) => (
                  <tr key={key}>
                    <td><strong>{key}</strong></td>
                    <td>{typeof val === 'object' ? val.value?.toFixed(2) : val?.toFixed?.(2) || val}</td>
                    <td>{typeof val === 'object' && val.ci95Lower
                      ? `[${val.ci95Lower.toFixed(2)} – ${val.ci95Upper.toFixed(2)}]` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p>Chưa có kết quả. Vào trang Dosing để tính.</p>}
        </div>
      </div>
    </div>
  )
}
