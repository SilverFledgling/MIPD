import { useState } from 'react'

export default function Settings() {
  const [model, setModel] = useState('v1.0')
  const [algo, setAlgo] = useState('Bayesian')
  const [role, setRole] = useState('Clinician')
  const [lang, setLang] = useState('vi')
  return (
    <div className="card form" style={{ maxWidth: 600 }}>
      <h2>Settings</h2>
      <label>Model version
        <select value={model} onChange={e => setModel(e.target.value)}>
          <option value="v1.0">v1.0</option>
          <option value="v1.1">v1.1</option>
          <option value="v2.0">v2.0</option>
        </select>
      </label>
      <label>Algorithm
        <select value={algo} onChange={e => setAlgo(e.target.value)}>
          <option>Bayesian</option>
          <option>ML</option>
        </select>
      </label>
      <label>User role
        <select value={role} onChange={e => setRole(e.target.value)}>
          <option>Admin</option>
          <option>Clinician</option>
        </select>
      </label>
      <label>Language
        <select value={lang} onChange={e => setLang(e.target.value)}>
          <option value="vi">Tiếng Việt</option>
          <option value="en">English</option>
        </select>
      </label>
      <button className="btn primary">Save</button>
    </div>
  )
}
