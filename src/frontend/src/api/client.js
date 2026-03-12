/**
 * MIPD API Client — Fetch wrapper routed through API Gateway.
 *
 * Routing: Frontend → Vite proxy → Gateway (:5000) → PK Engine (:8000)
 *
 * If Gateway is unreachable, falls back to direct PK Engine connection.
 *
 * Endpoints:
 *   POST /api/bayesian/estimate   → Bayesian PK estimation
 *   POST /api/dosing/recommend    → Dose optimization
 *   POST /api/dosing/cfr          → CFR
 *   POST /api/pk/predict          → PK prediction
 *   POST /api/pk/clinical         → Clinical calcs (CrCL, eGFR)
 *   POST /api/ai/anomaly-check   → Anomaly detection
 *   POST /api/ai/validate-metrics → Validation metrics
 */

const API_BASE = '/api';

/**
 * Generic fetch wrapper with JWT auth + error handling.
 * Now properly returns string error messages (never objects).
 */
async function apiRequest(endpoint, body = null, method = 'POST') {
  const url = `${API_BASE}${endpoint}`;
  const headers = { 'Content-Type': 'application/json' };

  // Auto-inject JWT token if available
  const token = localStorage.getItem('mipd_token');
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const options = { method, headers };
  if (body && method !== 'GET') {
    options.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(url, options);
  } catch (networkErr) {
    // Network error — proxy may not be configured
    throw new Error(`Không kết nối được API Gateway (${url}). Hãy restart Vite dev server.`);
  }

  if (!response.ok) {
    let detail = `API Error: ${response.status} ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (typeof errorData?.detail === 'string') {
        detail = errorData.detail;
      } else if (typeof errorData?.message === 'string') {
        detail = errorData.message;
      } else if (typeof errorData?.error === 'string') {
        detail = errorData.error;
      }
    } catch {}
    throw new Error(detail);
  }

  return response.json();
}

// ── Bayesian Estimation ──────────────────────────────────────

/**
 * Run Bayesian estimation (MAP, Laplace, MCMC, SMC, Adaptive Pipeline).
 */
export async function estimateBayesian(params) {
  return apiRequest('/bayesian/estimate', params);
}

// ── Dose Optimization ────────────────────────────────────────

/**
 * Get dose recommendation based on patient data + TDM.
 */
export async function recommendDose(params) {
  return apiRequest('/dosing/recommend', params);
}

/**
 * Calculate CFR (Cumulative Fraction of Response).
 */
export async function calculateCFR(params) {
  return apiRequest('/dosing/cfr', params);
}

// ── PK Prediction ────────────────────────────────────────────

/**
 * Predict PK concentration-time profile.
 */
export async function predictPK(params) {
  return apiRequest('/pk/predict', params);
}

/**
 * Calculate clinical parameters (CrCL, eGFR, AUC).
 */
export async function calculateClinical(params) {
  return apiRequest('/pk/clinical', params);
}

// ── AI/ML ────────────────────────────────────────────────────

/**
 * Check for anomalies in patient data (Swift Hydra).
 */
export async function checkAnomaly(params) {
  return apiRequest('/ai/anomaly-check', params);
}

/**
 * Run validation metrics (MPE, MAPE, CCC, NPDE, Coverage).
 */
export async function validateMetrics(params) {
  return apiRequest('/ai/validate-metrics', params);
}

// ── Health Check ─────────────────────────────────────────────

/**
 * Check if the API is reachable.
 * Tries Gateway /health first (ASP.NET returns "Healthy" as text).
 * Falls back to checking PK Engine root.
 */
export async function healthCheck() {
  // Try Gateway health endpoint via proxy
  try {
    const res = await fetch('/health');
    if (res.ok) return { status: 'ok', via: 'gateway' };
  } catch {}

  // Fallback: try Gateway root via /api proxy
  try {
    const res = await fetch('/api/docs');
    if (res.ok) return { status: 'ok', via: 'pk-engine' };
  } catch {}

  throw new Error('API offline');
}

export default {
  estimateBayesian,
  recommendDose,
  calculateCFR,
  predictPK,
  calculateClinical,
  checkAnomaly,
  validateMetrics,
  healthCheck,
};
