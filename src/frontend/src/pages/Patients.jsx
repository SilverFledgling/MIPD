import { useMemo, useState } from 'react'

const initialPatients = [
  { id: 1, name: 'Nguyễn A', age: 65, sex: 'male', weight: 70, height: 168, scr: 1.0, diagnosis: 'Pneumonia' },
  { id: 2, name: 'Trần B', age: 54, sex: 'female', weight: 55, height: 160, scr: 0.9, diagnosis: 'Sepsis' },
  { id: 3, name: 'Lê C', age: 72, sex: 'male', weight: 80, height: 170, scr: 1.3, diagnosis: 'Endocarditis' }
]

export default function Patients() {
  const [query, setQuery] = useState('')
  const [sexFilter, setSexFilter] = useState('all')
  const [page, setPage] = useState(1)
  const [rows, setRows] = useState(initialPatients)
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState({ id: null, name: '', age: '', sex: 'female', weight: '', height: '', scr: '', diagnosis: '' })
  const pageSize = 5

  const filtered = useMemo(() => {
    return rows.filter(p =>
      p.name.toLowerCase().includes(query.toLowerCase()) &&
      (sexFilter === 'all' || p.sex === sexFilter)
    )
  }, [rows, query, sexFilter])
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
  const pageData = filtered.slice((page - 1) * pageSize, page * pageSize)

  function openAdd() {
    setForm({ id: null, name: '', age: '', sex: 'female', weight: '', height: '', scr: '', diagnosis: '' })
    setShowForm(true)
  }
  function openEdit(p) {
    setForm({ ...p })
    setShowForm(true)
  }
  function remove(id) {
    setRows(r => r.filter(x => x.id !== id))
  }
  function onChange(e) {
    const { name, value } = e.target
    setForm(f => ({ ...f, [name]: value }))
  }
  function onSubmit(e) {
    e.preventDefault()
    if (form.id == null) {
      const id = Math.max(0, ...rows.map(r => r.id)) + 1
      setRows(r => [...r, { ...form, id }])
    } else {
      setRows(r => r.map(x => x.id === form.id ? { ...form } : x))
    }
    setShowForm(false)
  }

  return (
    <div className="grid">
      <div className="card">
        <h2>Patients</h2>
        <div className="form" style={{ gridTemplateColumns: '1fr 200px auto', alignItems: 'end' }}>
          <label>Search
            <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Enter patient name" />
          </label>
          <label>Filter
            <select value={sexFilter} onChange={e => setSexFilter(e.target.value)}>
              <option value="all">All</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </label>
          <button className="btn primary" onClick={openAdd}>Add Patient</button>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="table">
            <thead>
              <tr>
                <th>Name</th><th>Age</th><th>Sex</th><th>Weight</th><th>Height</th><th>SCr</th><th>Diagnosis</th><th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {pageData.map(p => (
                <tr key={p.id}>
                  <td>{p.name}</td>
                  <td>{p.age}</td>
                  <td>{p.sex}</td>
                  <td>{p.weight}</td>
                  <td>{p.height}</td>
                  <td>{p.scr}</td>
                  <td>{p.diagnosis}</td>
                  <td>
                    <button className="btn" onClick={() => openEdit(p)}>Edit</button>
                    <button className="btn" onClick={() => remove(p.id)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <button className="btn" onClick={() => setPage(p => Math.max(1, p - 1))}>Prev</button>
          <div style={{ alignSelf: 'center' }}>Page {page} / {totalPages}</div>
          <button className="btn" onClick={() => setPage(p => Math.min(totalPages, p + 1))}>Next</button>
        </div>
      </div>

      {showForm && (
        <form className="card form" onSubmit={onSubmit}>
          <h3>{form.id == null ? 'Add Patient' : 'Edit Patient'}</h3>
          <label>Name <input name="name" value={form.name} onChange={onChange} required /></label>
          <label>Age <input name="age" type="number" value={form.age} onChange={onChange} required /></label>
          <label>Sex
            <select name="sex" value={form.sex} onChange={onChange}>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </label>
          <label>Weight <input name="weight" type="number" value={form.weight} onChange={onChange} required /></label>
          <label>Height <input name="height" type="number" value={form.height} onChange={onChange} required /></label>
          <label>Serum Creatinine <input name="scr" type="number" step="0.01" value={form.scr} onChange={onChange} required /></label>
          <label>Diagnosis <input name="diagnosis" value={form.diagnosis} onChange={onChange} /></label>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn primary" type="submit">Save</button>
            <button className="btn" type="button" onClick={() => setShowForm(false)}>Cancel</button>
          </div>
        </form>
      )}
    </div>
  )
}
