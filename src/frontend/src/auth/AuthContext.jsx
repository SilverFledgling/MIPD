/**
 * MIPD Auth Context — JWT authentication state management.
 *
 * Provides: login, logout, register, current user, token.
 * Uses localStorage for token persistence.
 *
 * Theo thuyết minh CV 7.3: OAuth2/JWT, RBAC 4 roles
 */
import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

const API_BASE = '/api';

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('mipd_token'));
  const [loading, setLoading] = useState(true);

  // On mount: verify stored token
  useEffect(() => {
    if (token) {
      fetchMe(token)
        .then(setUser)
        .catch(() => { setToken(null); localStorage.removeItem('mipd_token'); })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  async function login(email, password) {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) throw new Error('Invalid credentials');
    const data = await res.json();
    setToken(data.token);
    setUser(data);
    localStorage.setItem('mipd_token', data.token);
    return data;
  }

  async function register(email, password, fullName, role) {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, fullName, role }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Registration failed');
    }
    const data = await res.json();
    setToken(data.token);
    setUser(data);
    localStorage.setItem('mipd_token', data.token);
    return data;
  }

  function logout() {
    setToken(null);
    setUser(null);
    localStorage.removeItem('mipd_token');
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

async function fetchMe(token) {
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error('Token invalid');
  return res.json();
}
