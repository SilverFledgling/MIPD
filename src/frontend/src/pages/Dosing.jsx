import { useState } from 'react'
import { recommendDose, estimateBayesian, calculateClinical } from '../api/client'

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
      // Build request body matching API schema
      const hasTDM = form.conc && form.sampleTime
      const isPedi = form.model === 'vancomycin_pedi'
      const body = {
        patient: {
          age: +form.age,
          weight: +form.weight,
          height: +form.height || 165,
          gender: form.sex,
          serum_creatinine: +form.scr,
          ...(isPedi && form.pma ? { pma: +form.pma } : {}),
        },
        model: form.model,
        dose: [{
          time: +form.doseTime || 0,
          amount: +form.dose || 1000,
          duration: +form.infusion || 1,
        }],
        method: form.method,
        target: {
          auc24_min: 400,
          auc24_max: 600,
          mic: +form.mic,
        },
      }

      if (hasTDM) {
        body.observations = [{
          time: +form.sampleTime,
          concentration: +form.conc,
        }]
      }

      // Call Bayesian estimation API
      const result = await estimateBayesian(body)
      setRes(result)

      // Save to localStorage for Results page
      try { localStorage.setItem('vanco_result', JSON.stringify(result)) } catch {}
    } catch (err) {
      setError(err.message)
      // Fallback to local calculation if API unavailable
      const crcl = ((140 - (+form.age)) * (+form.weight)) / (72 * (+form.scr)) * (form.sex === 'female' ? 0.85 : 1)
      const cl = Math.max(1, Math.min(9, crcl / 10))
      const aucNeed = (+form.auc) * (+form.mic)
      const daily = aucNeed * cl
      const interval = daily > 2500 ? 8 : daily > 1600 ? 12 : 24
      const perDose = Math.round(daily / (24 / interval) / 50) * 50
      setRes({
        fallback: true,
        recommendation: { dose: perDose, interval },
        predictions: { auc24: Math.round((perDose * (24 / interval)) / cl) },
        individualParams: { CL: { value: +cl.toFixed(2) } },
      })
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
              <option value="mcmc_nuts">MCMC-NUTS (JAX/NumPyro — cần gradient)</option>
              <option value="mcmc_mh">MCMC-MH (Metropolis-Hastings — thuần Python)</option>
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
          <input name="weight" type="number" min="2" max="300" step="0.1" value={form.weight} onChange={onChange} placeholder="ví dụ 70" required />
        </label>
        <label>Chiều cao (cm)
          <input name="height" type="number" min="30" max="230" value={form.height} onChange={onChange} placeholder="ví dụ 168" />
        </label>
        <label>Giới tính
          <select name="sex" value={form.sex} onChange={onChange}>
            <option value="female">Nữ</option>
            <option value="male">Nam</option>
          </select>
        </label>
        <label>Creatinine huyết thanh (mg/dL)
          <input name="scr" type="number" min="0.2" max="15" step="0.01" value={form.scr} onChange={onChange} placeholder="ví dụ 1.0" required />
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
            <ul>
              <li>Liều đề xuất: <strong>{res.recommendation?.dose || '—'} mg</strong></li>
              <li>Khoảng cách liều: q{res.recommendation?.interval || '—'}h</li>
              <li>AUC24 ước tính: {res.predictions?.auc24 || '—'} mg·h/L</li>
              {res.individualParams?.CL && <li>CL: {res.individualParams.CL.value || res.individualParams.CL} L/h</li>}
              {res.method && <li>Phương pháp: {res.method}</li>}
              {res.diagnostics?.layers_executed && (
                <li>Pipeline layers: {res.diagnostics.layers_executed.join(' → ')}</li>
              )}
            </ul>
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
