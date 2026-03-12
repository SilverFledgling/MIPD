/**
 * MIPD API Client — Fetch wrapper routed through API Gateway.
 *
 * Replaces all mock data with real API calls.
 * API Base URL: /api → Vite proxy → Gateway (:5000) → PK Engine (:8000)
 * JWT token auto-injected from localStorage.
 *
 * Endpoints:
 *   POST /api/bayesian/estimate   → Bayesian PK estimation
 *   POST /api/dosing/recommend    → Dose optimization
 *   POST /api/dosing/cfr          → Cumulative Fraction of Response
 *   POST /api/pk/predict          → PK concentration prediction
 *   POST /api/pk/clinical         → Clinical calculations (CrCL, eGFR)
 *   POST /api/ai/anomaly-check   → Swift Hydra anomaly detection
 *   POST /api/ai/validate-metrics → Validation metrics (MPE, MAPE, etc.)
 */

const API_BASE = '/api';

/**
 * Generic fetch wrapper with JWT auth + error handling.
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

  const response = await fetch(url, options);

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `API Error: ${response.status} ${response.statusText}`
    );
  }

  return response.json();
}

// ── Bayesian Estimation ──────────────────────────────────────

/**
 * Run Bayesian estimation (MAP, Laplace, MCMC, SMC, Adaptive Pipeline).
 *
 * @param {Object} params - { patient, dose, observations, method, target }
 * @returns {Promise<Object>} BayesianResponse
 */
export async function estimateBayesian(params) {
  return apiRequest('/bayesian/estimate', params);
}

// ── Dosing ───────────────────────────────────────────────────

/**
 * Get dose recommendation based on patient data and PK model.
 *
 * @param {Object} params - { patient, doses, observations, target }
 * @returns {Promise<Object>} DosingResponse
 */
export async function recommendDose(params) {
  return apiRequest('/dosing/recommend', params);
}

/**
 * Calculate Cumulative Fraction of Response (CFR).
 *
 * @param {Object} params - { auc_values, mic_distribution }
 * @returns {Promise<Object>} CFRResponse
 */
export async function calculateCFR(params) {
  return apiRequest('/dosing/cfr', params);
}

// ── PK Prediction ────────────────────────────────────────────

/**
 * Predict PK concentrations over time.
 *
 * @param {Object} params - { patient, doses, times, model_type }
 * @returns {Promise<Object>} PKPredictionResponse
 */
export async function predictPK(params) {
  return apiRequest('/pk/predict', params);
}

/**
 * Calculate clinical values (CrCL, eGFR, BMI, etc.).
 *
 * @param {Object} patient - Patient demographics
 * @returns {Promise<Object>} ClinicalResponse
 */
export async function calculateClinical(patient) {
  return apiRequest('/pk/clinical', patient);
}

// ── AI/ML ────────────────────────────────────────────────────

/**
 * Run anomaly detection (Swift Hydra) on TDM sample.
 *
 * @param {Object} params - { concentration, predicted, patient_history }
 * @returns {Promise<Object>} AnomalyResponse
 */
export async function checkAnomaly(params) {
  return apiRequest('/ai/anomaly-check', params);
}

/**
 * Run validation metrics (MPE, MAPE, CCC, NPDE, etc.).
 *
 * @param {Object} params - { estimated, true_values }
 * @returns {Promise<Object>} ValidationMetricsResponse
 */
export async function validateMetrics(params) {
  return apiRequest('/ai/validate-metrics', params);
}

// ── Health Check ─────────────────────────────────────────────

/**
 * Check if the PK Engine API is reachable.
 *
 * @returns {Promise<Object>} { status: 'ok' }
 */
export async function healthCheck() {
  return apiRequest('/health', null, 'GET');
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
