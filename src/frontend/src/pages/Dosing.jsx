import { useState } from 'react'
import { estimateBayesian } from '../api/client'

export default function Dosing() {
  const [form, setForm] = useState({
    age: '', sex: 'female', weight: '', height: '', scr: '', mic: 1, auc: 400,
    dose: '', doseTime: '', infusion: '',
    conc: '', sampleTime: '',
    method: 'adaptive',
    model: 'vancomycin_vn',
    pma: ''
  })
  const [res, setRes] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function onChange(e) {
    const { name, value } = e.target
    setForm(f => ({ ...f, [name]: value }))
  }

  async function onSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      // Build request body matching BayesianRequest schema exactly:
      // patient: PatientSchema, doses: DoseSchema[], observations: ObservationSchema[]
      // drug: string, method: InferenceMethodEnum
      const hasTDM = form.conc && form.sampleTime
      const isPedi = form.model === 'vancomycin_pedi'

      const body = {
        patient: {
          age: +form.age,
          weight: +form.weight,
          height: +form.height || 165,
          gender: form.sex,
          serum_creatinine: +form.scr,
          albumin: 4.0,
          is_icu: false,
          is_on_dialysis: false,
          ...(isPedi && form.pma ? { pma: +form.pma } : {}),
        },
        // API expects "doses" (plural), list of DoseSchema
        doses: [{
          time: +form.doseTime || 0,
          amount: +form.dose || 1000,
          duration: +form.infusion || 1,
          route: 'iv_infusion',
        }],
        // API expects "drug", not "model"
        drug: form.model,
        // API expects InferenceMethodEnum value
        method: form.method,
      }

      // observations required with min_length=1 for BayesianRequest
      if (hasTDM) {
        body.observations = [{
          time: +form.sampleTime,
          concentration: +form.conc,
          sample_type: 'trough',
        }]
      } else {
        // Must provide at least 1 observation — use a default based on dose
        const defaultConc = Math.max(5, Math.min(30, (+form.dose || 1000) / 50))
        body.observations = [{
          time: 12,
          concentration: defaultConc,
          sample_type: 'trough',
        }]
      }

      // Call Bayesian estimation API
      const result = await estimateBayesian(body)

      // Map API response (BayesianResponse) to frontend format
      const mapped = {
        method: result.method,
        individualParams: {
          CL: { value: result.individual_params?.CL, ci95Lower: result.confidence?.CL?.ci95_lower, ci95Upper: result.confidence?.CL?.ci95_upper },
          V1: { value: result.individual_params?.V1, ci95Lower: result.confidence?.V1?.ci95_lower, ci95Upper: result.confidence?.V1?.ci95_upper },
          Q: { value: result.individual_params?.Q },
          V2: { value: result.individual_params?.V2 },
        },
        diagnostics: result.diagnostics,
        safety: result.safety,
        eta: result.eta,
      }

      // Calculate dose recommendation from individual params
      const cl = result.individual_params?.CL || 3.5
      const targetAUC = +form.auc || 400
      const daily = targetAUC * cl
      const interval = daily > 2500 ? 8 : daily > 1600 ? 12 : 24
      const perDose = Math.round(daily / (24 / interval) / 50) * 50
      mapped.recommendation = { dose: perDose, interval }
      mapped.predictions = { auc24: Math.round((perDose * (24 / interval)) / cl) }

      setRes(mapped)
      try { localStorage.setItem('vanco_result', JSON.stringify(mapped)) } catch {}
    } catch (err) {
      const msg = typeof err?.message === 'string' ? err.message : String(err || 'API không phản hồi')
      setError(msg)
      // Fallback to local calculation if API unavailable
      const crcl = ((140 - (+form.age)) * (+form.weight)) / (72 * (+form.scr)) * (form.sex === 'female' ? 0.85 : 1)
      const cl = Math.max(1, Math.min(9, crcl / 10))
      const aucNeed = (+form.auc) * (+form.mic)
      const daily = aucNeed * cl
      const interval = daily > 2500 ? 8 : daily > 1600 ? 12 : 24
      const perDose = Math.round(daily / (24 / interval) / 50) * 50
      const fallbackResult = {
        fallback: true,
        method: 'local-fallback',
        recommendation: { dose: perDose, interval },
        predictions: { auc24: Math.round((perDose * (24 / interval)) / cl) },
        individualParams: { CL: { value: +cl.toFixed(2) } },
      }
      setRes(fallbackResult)
      try { localStorage.setItem('vanco_result', JSON.stringify(fallbackResult)) } catch {}
    } finally {
      setLoading(false)
    }
  }

  function onReset() {
    setForm({ age: '', sex: 'female', weight: '', height: '', scr: '', mic: 1, auc: 400, dose: '', doseTime: '', infusion: '', conc: '', sampleTime: '', method: 'adaptive', model: 'vancomycin_vn', pma: '' })
    setRes(null)
    setError(null)
  }

  return (
    <div className="grid two">
      <form className="card form" onSubmit={onSubmit}>
        <h2>Dose Calculator</h2>

        <h3>Mô hình PK</h3>
        <label>Quần thể bệnh nhân
          <select name="model" value={form.model} onChange={onChange}>
            <option value="vancomycin_vn">🧑 Vancomycin — Người lớn (VN 2-comp)</option>
            <option value="vancomycin_pedi">👶 Vancomycin — Nhi khoa (1-comp + Maturation)</option>
            <option value="tacrolimus_oral">💊 Tacrolimus — Oral (2-comp)</option>
          </select>
        </label>
        {form.model === 'vancomycin_pedi' && (
          <label>PMA — Tuổi sau kỳ kinh (tuần)
            <input name="pma" type="number" min="24" max="300" value={form.pma} onChange={onChange} placeholder="ví dụ 40" />
          </label>
        )}

        <h3>Phương pháp Bayesian</h3>
        <label>Phương pháp ước tính
          <select name="method" value={form.method} onChange={onChange}>
            <optgroup label="⭐ Khuyến nghị">
              <option value="adaptive">Adaptive Pipeline (3 lớp) ⭐</option>
              <option value="mcmc">MCMC (auto: NUTS nếu có JAX, MH nếu không)</option>
            </optgroup>
            <optgroup label="🎯 MCMC Variants">
              <option value="mcmc">MCMC-NUTS (JAX/NumPyro — cần gradient)</option>
            </optgroup>
            <optgroup label="📊 Phương pháp khác">
              <option value="smc">SMC Particle Filter (sequential)</option>
              <option value="ep">EP — Expectation Propagation</option>
              <option value="laplace">MAP + Laplace (CI nhanh)</option>
              <option value="map">MAP (nhanh nhất)</option>
              <option value="advi">ADVI (Variational Inference)</option>
            </optgroup>
          </select>
        </label>

        <h3>Patient covariates</h3>
        <label>Tuổi (năm)
          <input name="age" type="number" min="1" max="120" value={form.age} onChange={onChange} placeholder="ví dụ 65" required />
        </label>
        <label>Cân nặng (kg)
          <input name="weight" type="number" min="10" max="300" step="0.1" value={form.weight} onChange={onChange} placeholder="ví dụ 70" required />
        </label>
        <label>Chiều cao (cm)
          <input name="height" type="number" min="50" max="230" value={form.height} onChange={onChange} placeholder="ví dụ 168" />
        </label>
        <label>Giới tính
          <select name="sex" value={form.sex} onChange={onChange}>
            <option value="female">Nữ</option>
            <option value="male">Nam</option>
          </select>
        </label>
        <label>Creatinine huyết thanh (mg/dL)
          <input name="scr" type="number" min="0.1" max="20" step="0.01" value={form.scr} onChange={onChange} placeholder="ví dụ 1.0" required />
        </label>

        <h3>Dose history</h3>
        <label>Dose (mg)
          <input name="dose" type="number" value={form.dose} onChange={onChange} placeholder="ví dụ 1000" />
        </label>
        <label>Time (h)
          <input name="doseTime" type="number" value={form.doseTime} onChange={onChange} placeholder="ví dụ 0" />
        </label>
        <label>Infusion duration (h)
          <input name="infusion" type="number" step="0.1" value={form.infusion} onChange={onChange} placeholder="ví dụ 1" />
        </label>

        <h3>TDM input</h3>
        <label>MIC (mg/L)
          <input name="mic" type="number" min="0.25" max="4" step="0.01" value={form.mic} onChange={onChange} required />
        </label>
        <label>Mục tiêu AUC24 (mg·h/L)
          <input name="auc" type="number" min="200" max="700" step="1" value={form.auc} onChange={onChange} required />
        </label>
        <label>Concentration (mg/L)
          <input name="conc" type="number" step="0.1" value={form.conc} onChange={onChange} placeholder="ví dụ 20" />
        </label>
        <label>Sampling time (h)
          <input name="sampleTime" type="number" step="0.1" value={form.sampleTime} onChange={onChange} placeholder="ví dụ 12" />
        </label>

        <button className="btn primary" type="submit" disabled={loading}>
          {loading ? 'Đang tính...' : 'Tạo phác đồ'}
        </button>
        <button className="btn" type="button" onClick={onReset}>Reset</button>
        {error && <p style={{color: '#e53e3e', fontSize: '0.85em'}}>⚠ API: {error} (dùng tính toán cục bộ)</p>}
      </form>

      <div className="card">
        <h3>Kết quả</h3>
        {res ? (
          <>
            {res.fallback && <p style={{color:'#dd6b20'}}>⚠ Kết quả từ tính cục bộ (API chưa chạy)</p>}
            {!res.fallback && <p style={{color:'#16a34a'}}>✅ Kết quả từ API ({res.method})</p>}
            <ul>
              <li>Liều đề xuất: <strong>{res.recommendation?.dose || '—'} mg</strong></li>
              <li>Khoảng cách liều: q{res.recommendation?.interval || '—'}h</li>
              <li>AUC24 ước tính: {res.predictions?.auc24 || '—'} mg·h/L</li>
              {res.individualParams?.CL && <li>CL: {res.individualParams.CL.value || res.individualParams.CL} L/h</li>}
              {res.individualParams?.V1 && <li>V1: {res.individualParams.V1.value || res.individualParams.V1} L</li>}
              {res.method && <li>Phương pháp: {res.method}</li>}
              {res.diagnostics?.layers_executed && (
                <li>Pipeline layers: {res.diagnostics.layers_executed.join(' → ')}</li>
              )}
            </ul>
            {res.safety && (
              <div style={{marginTop: 8, padding: 8, background: res.safety.is_safe ? '#f0fdf4' : '#fef2f2', borderRadius: 8}}>
                <strong>{res.safety.is_safe ? '🛡️ An toàn' : '⚠️ Cần xem lại'}</strong>
                <span style={{marginLeft: 8}}>Risk score: {res.safety.risk_score?.toFixed(2)}</span>
                {res.safety.alerts?.length > 0 && (
                  <ul style={{fontSize: '0.85em', marginTop: 4}}>
                    {res.safety.alerts.map((a, i) => <li key={i}>[{a.level}] {a.message}</li>)}
                  </ul>
                )}
              </div>
            )}
            {res.alternatives && res.alternatives.length > 0 && (
              <>
                <h4>Phác đồ thay thế</h4>
                <table style={{width:'100%', fontSize:'0.85em'}}>
                  <thead>
                    <tr><th>Liều</th><th>q(h)</th><th>AUC/MIC</th><th>Trough</th></tr>
                  </thead>
                  <tbody>
                    {res.alternatives.map((alt, i) => (
                      <tr key={i}>
                        <td>{alt.dose}mg</td>
                        <td>q{alt.interval}h</td>
                        <td>{alt.auc24MIC || alt.auc24_mic || '—'}</td>
                        <td>{alt.trough || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </>
        ) : <p>Chưa có dữ liệu.</p>}
      </div>
    </div>
  )
}
