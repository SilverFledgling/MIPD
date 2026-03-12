import { NavLink, Route, Routes, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './auth/AuthContext'
import Dashboard from './pages/Dashboard'
import Patients from './pages/Patients'
import Dosing from './pages/Dosing'
import Results from './pages/Results'
import Validation from './pages/Validation'
import Settings from './pages/Settings'
import Login from './pages/Login'
import Topbar from './components/Topbar'

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="loading">Loading...</div>;
  return user ? children : <Navigate to="/login" />;
}

function AppLayout() {
  const { user, logout } = useAuth();

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand">VancoDose</div>
        <nav className="sidebar-nav">
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/patients">Patients</NavLink>
          <NavLink to="/dosing">Dose Calculator</NavLink>
          <NavLink to="/results">Results</NavLink>
          <NavLink to="/validation">Validation</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </nav>
        {user && (
          <div className="sidebar-user">
            <small>{user.fullName}</small>
            <small className="role-badge">{user.role}</small>
            <button onClick={logout} className="btn-logout">Đăng xuất</button>
          </div>
        )}
      </aside>
      <main className="content">
        <Topbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/patients" element={<Patients />} />
          <Route path="/dosing" element={<Dosing />} />
          <Route path="/results" element={<Results />} />
          <Route path="/validation" element={<Validation />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/*" element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        } />
      </Routes>
    </AuthProvider>
  );
}
