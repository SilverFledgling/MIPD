import { useState, useEffect } from 'react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, AreaChart, Area, LineChart, Line, PieChart, Pie, Cell, Legend,
} from 'recharts'
import { healthCheck } from '../api/client'

const COLORS = ['#0ea5e9', '#14b8a6', '#f59e0b', '#ef4444', '#8b5cf6']

export default function Dashboard() {
  const [apiStatus, setApiStatus] = useState('checking')
  const [storeInfo, setStoreInfo] = useState(null)
  const [recentResults, setRecentResults] = useState([])

  useEffect(() => {
    // Check API health
    healthCheck()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'))

    // Load recent results from localStorage
    try {
      const stored = JSON.parse(localStorage.getItem('vanco_result') || 'null')
      if (stored) setRecentResults([stored])
    } catch {}

    // Load population store stats
    try {
      const store = JSON.parse(localStorage.getItem('population_store_info') || 'null')
      if (store) setStoreInfo(store)
    } catch {}
  }, [])

  // Summary cards
  const cards = [
    {
      label: 'API Status',
      value: apiStatus === 'online' ? '✅ Online' : apiStatus === 'offline' ? '❌ Offline' : '⏳ Checking...',
      color: apiStatus === 'online' ? '#22c55e' : '#ef4444',
    },
    {
      label: 'Mô hình PK',
      value: '3',
      sub: 'Người lớn + Nhi khoa + Tacrolimus',
    },
    {
      label: 'Phương pháp Bayesian',
      value: '8',
      sub: 'MAP, Laplace, MCMC, ADVI, EP, SMC, BMA, Adaptive',
    },
    {
      label: 'Tier 2 (VN Population)',
      value: storeInfo?.n_individuals || '0',
      sub: 'Bệnh nhân đã cập nhật θ_VN',
    },
  ]

  // Method distribution pie chart
  const methodDist = [
    { name: 'Adaptive (3 lớp)', value: 45 },
    { name: 'MAP', value: 25 },
    { name: 'MCMC/NUTS', value: 15 },
    { name: 'SMC', value: 10 },
    { name: 'Laplace', value: 5 },
  ]

  // AUC distribution histogram
  const aucData = Array.from({ length: 20 }).map((_, i) => ({
    bin: `${200 + i * 25}`,
    count: Math.round(Math.max(1, 18 - Math.abs(8 - i) * 1.8)),
    inTarget: (200 + i * 25) >= 400 && (200 + i * 25) <= 600,
  }))

  // Patients per day trend
  const patientsDaily = Array.from({ length: 14 }).map((_, i) => ({
    day: `D${i + 1}`,
    count: Math.round(10 + Math.random() * 15),
    alerts: Math.round(Math.random() * 4),
  }))

  return (
    <div className="grid">
      {/* Summary Cards */}
      <div className="grid two">
        {cards.map(c => (
          <div className="card" key={c.label}>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
              <div style={{ color: '#475569', fontSize: 13 }}>{c.label}</div>
            </div>
            <div style={{ fontSize: 28, fontWeight: 800, color: c.color || '#1e293b' }}>{c.value}</div>
            {c.sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{c.sub}</div>}
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid two">
        <div className="card">
          <h3>Phân bố AUC₂₄/MIC (bệnh nhân ảo)</h3>
          <p style={{ fontSize: '0.8em', color: '#666' }}>
            <span style={{ color: '#22c55e' }}>■</span> Trong vùng đích (400–600) &nbsp;|&nbsp;
            <span style={{ color: '#94a3b8' }}>■</span> Ngoài vùng đích
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={aucData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin" fontSize={11} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count">
                {aucData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.inTarget ? '#22c55e' : '#cbd5e1'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Phương pháp sử dụng phổ biến</h3>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={methodDist} dataKey="value" nameKey="name" cx="50%" cy="50%"
                   outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {methodDist.map((_, idx) => (
                  <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Trend + 3-Tier Info */}
      <div className="grid two">
        <div className="card">
          <h3>Số bệnh nhân & Cảnh báo theo ngày</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={patientsDaily}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#14b8a6" name="Bệnh nhân" />
              <Bar dataKey="alerts" fill="#ef4444" name="Cảnh báo" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Mô hình Phân cấp 3 Tầng</h3>
          <div style={{ fontSize: '0.9em', lineHeight: 1.8 }}>
            <div style={{ padding: '8px 12px', background: '#eff6ff', borderRadius: 6, marginBottom: 6 }}>
              <strong>Tier 1 — Global (Quốc tế)</strong><br/>
              CL = 4.5 L/h, V₁ = 30 L &nbsp; <span style={{color:'#94a3b8'}}>| Goti 2018 (CỐ ĐỊNH)</span>
            </div>
            <div style={{ padding: '8px 12px', background: '#f0fdf4', borderRadius: 6, marginBottom: 6 }}>
              <strong>Tier 2 — Vietnam (Nội địa)</strong><br/>
              CL = {storeInfo?.mu_CL?.toFixed(2) || '—'} L/h, V₁ = {storeInfo?.mu_V1?.toFixed(1) || '—'} L
              &nbsp; <span style={{color:'#22c55e'}}>| n = {storeInfo?.n_individuals || 0} (CẬP NHẬT LIÊN TỤC)</span>
            </div>
            <div style={{ padding: '8px 12px', background: '#fefce8', borderRadius: 6 }}>
              <strong>Tier 3 — Individual (Cá thể)</strong><br/>
              θ_i ← MAP/Laplace/SMC/MCMC &nbsp; <span style={{color:'#f59e0b'}}>| Ước tính riêng từng BN</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
