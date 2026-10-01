/**
 * Cardiometabolic Digital Twin — Clinical Dashboard Application
 * Handles API calls, real-time patient rendering, Chart.js multi-signal timelines,
 * SHAP explainability visualizations, and What-If counterfactual simulations.
 */

// 1. Constants and Configuration
const API_BASE = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
    ? window.location.origin
    : 'http://localhost:8000';

const GLUCOSE_ZONES = {
    critical_high: 250,
    high: 180,
    target_high: 180,
    target_low: 70,
    low: 54,
};

const RISK_COLORS = {
    low: '#22c55e',
    moderate: '#f59e0b',
    high: '#ef4444',
    critical: '#dc2626',
};

const STREAM_COLORS = {
    static: '#3b82f6',   // blue
    dynamic: '#f97316',  // orange
};

// 2. State Management
let state = {
    currentPatientId: null,
    patientProfile: null,
    timeline: null,
    prediction: null,
    simulation: null,
    timelineChart: null,      // Chart.js instance
    scenarioCharts: [],       // Array of Chart.js instances
};

// 3. API Communication
async function fetchJSON(url, options = {}) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API Error: ${url}`, error);
        throw error;
    }
}

async function loadPatientList() {
    return await fetchJSON(`${API_BASE}/api/patients`);
}

async function loadPatientProfile(patientId) {
    return await fetchJSON(`${API_BASE}/api/patient/${patientId}`);
}

async function loadTimeline(patientId, hours = 12) {
    return await fetchJSON(`${API_BASE}/api/patient/${patientId}/timeline?hours=${hours}`);
}

async function loadPrediction(patientId) {
    return await fetchJSON(`${API_BASE}/api/patient/${patientId}/predict`);
}

async function loadSimulation(patientId) {
    return await fetchJSON(`${API_BASE}/api/patient/${patientId}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });
}

// 4. Formatting Utilities
function formatTime(timestamp) {
    if (!timestamp) return '';
    const parts = timestamp.split(' ');
    if (parts.length > 1) return parts[1].substring(0, 5);
    return timestamp;
}

function getRiskBadgeHTML(riskLevel) {
    const level = (riskLevel || 'low').toLowerCase();
    const colors = { low: '#22c55e', moderate: '#f59e0b', high: '#ef4444', critical: '#dc2626' };
    const icons = { low: '✅', moderate: '⚡', high: '⚠️', critical: '🚨' };
    const bg = colors[level] || '#22c55e';
    const textCol = level === 'moderate' ? '#000' : '#fff';
    return `<span class="badge badge-${level}" style="background:${bg};color:${textCol};">${icons[level] || '•'} ${level.toUpperCase()}</span>`;
}

function showLoading() {
    const el = document.getElementById('loading-overlay');
    if (el) el.style.display = 'flex';
}

function hideLoading() {
    const el = document.getElementById('loading-overlay');
    if (el) el.style.display = 'none';
}

// 5. Render Functions

/**
 * Render Section 1: Patient 360° Profile
 */
function renderPatientCard(profile) {
    const container = document.getElementById('patient-card');
    const badgeId = document.getElementById('patient-badge-id');
    if (badgeId) badgeId.textContent = profile.patient_id;

    // Interpretations
    let hba1cClass = 'pill-good';
    let hba1cText = 'Well controlled';
    if (profile.hba1c > 9.0) { hba1cClass = 'pill-crit'; hba1cText = 'Very poor'; }
    else if (profile.hba1c > 8.0) { hba1cClass = 'pill-bad'; hba1cText = 'Poor control'; }
    else if (profile.hba1c >= 7.0) { hba1cClass = 'pill-suboptimal'; hba1cText = 'Suboptimal'; }

    let egfrClass = 'pill-good';
    let egfrText = 'Normal';
    if (profile.egfr < 30) { egfrClass = 'pill-crit'; egfrText = 'Severe CKD'; }
    else if (profile.egfr < 60) { egfrClass = 'pill-bad'; egfrText = 'Moderate CKD'; }
    else if (profile.egfr < 90) { egfrClass = 'pill-suboptimal'; egfrText = 'Mild CKD'; }

    let bmiClass = 'pill-good';
    let bmiText = 'Normal';
    if (profile.bmi > 35) { bmiClass = 'pill-crit'; bmiText = 'Obese II'; }
    else if (profile.bmi >= 30) { bmiClass = 'pill-bad'; bmiText = 'Obese I'; }
    else if (profile.bmi >= 25) { bmiClass = 'pill-suboptimal'; bmiText = 'Overweight'; }

    const prsPct = Math.min(100, Math.max(0, profile.genetic_risk_score * 100));
    let prsColor = '#22c55e';
    if (prsPct > 66) prsColor = '#ef4444';
    else if (prsPct > 33) prsColor = '#f59e0b';

    container.innerHTML = `
        <div class="profile-meta-grid">
            <div class="meta-item">
                <span class="meta-label">Demographics</span>
                <span class="meta-value">${profile.age} yrs &bull; ${profile.sex === 'M' ? 'Male' : 'Female'}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Body Mass Index</span>
                <span class="meta-value">${profile.bmi} <span class="pill-badge ${bmiClass}">${bmiText}</span></span>
            </div>
            <div class="meta-item">
                <span class="meta-label">HbA1c Biomarker</span>
                <span class="meta-value">${profile.hba1c}% <span class="pill-badge ${hba1cClass}">${hba1cText}</span></span>
            </div>
            <div class="meta-item">
                <span class="meta-label">eGFR (Kidney)</span>
                <span class="meta-value">${profile.egfr} <span class="pill-badge ${egfrClass}">${egfrText}</span></span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Fasting Glucose</span>
                <span class="meta-value">${profile.fasting_glucose} mg/dL</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Systolic BP</span>
                <span class="meta-value">${profile.systolic_bp} mmHg</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">T2DM Duration</span>
                <span class="meta-value">${profile.years_with_t2dm} years</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Comorbidities</span>
                <span class="meta-value">${profile.comorbidity_count} active</span>
            </div>
        </div>

        <div class="clinical-biomarkers">
            <span class="section-label">Active Pharmacotherapy</span>
            <div class="medications-container">
                <span class="med-pill ${profile.med_metformin ? 'active' : 'inactive'}">Metformin ${profile.med_metformin ? '✓' : '—'}</span>
                <span class="med-pill ${profile.med_insulin_basal ? 'active' : 'inactive'}">Basal Insulin ${profile.med_insulin_basal ? '✓' : '—'}</span>
                <span class="med-pill ${profile.med_insulin_rapid ? 'active' : 'inactive'}">Rapid Insulin ${profile.med_insulin_rapid ? '✓' : '—'}</span>
                <span class="med-pill ${profile.med_sglt2i ? 'active' : 'inactive'}">SGLT2i ${profile.med_sglt2i ? '✓' : '—'}</span>
                <span class="med-pill ${profile.med_glp1ra ? 'active' : 'inactive'}">GLP-1 RA ${profile.med_glp1ra ? '✓' : '—'}</span>
            </div>
            ${profile.insulin_total_daily_dose > 0 ? `<div style="font-size:11px;color:var(--text-secondary);margin-top:4px;">Total Daily Insulin: <strong>${profile.insulin_total_daily_dose} Units/day</strong></div>` : ''}

            <div class="genetic-risk-container">
                <div style="display:flex;justify-content:space-between;font-size:11px;">
                    <span style="color:var(--text-secondary);">Polygenic Risk Score (PRS)</span>
                    <strong style="color:${prsColor};">${(profile.genetic_risk_score).toFixed(3)}</strong>
                </div>
                <div class="risk-bar-track">
                    <div class="risk-bar-fill" style="width:${prsPct}%;background:${prsColor};"></div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Custom background zone plugin for Chart.js
 */
const zoneBackgroundPlugin = {
    id: 'zoneBackgroundPlugin',
    beforeDraw: (chart) => {
        const { ctx, chartArea, scales } = chart;
        if (!chartArea || !scales.yGlucose) return;

        const { left, right } = chartArea;
        const yGlucose = scales.yGlucose;

        const y250 = yGlucose.getPixelForValue(250);
        const y180 = yGlucose.getPixelForValue(180);
        const y70 = yGlucose.getPixelForValue(70);
        const y54 = yGlucose.getPixelForValue(54);
        const yTop = chartArea.top;
        const yBottom = chartArea.bottom;

        ctx.save();

        // 1. Critical High Zone (> 250)
        if (y250 > yTop) {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.07)';
            ctx.fillRect(left, yTop, right - left, y250 - yTop);
        }

        // 2. Target In-Range Zone (70 - 180)
        if (y70 > y180) {
            ctx.fillStyle = 'rgba(34, 197, 94, 0.08)';
            ctx.fillRect(left, y180, right - left, y70 - y180);
        }

        // 3. Hypoglycemic Zone (< 70)
        if (yBottom > y70) {
            ctx.fillStyle = 'rgba(245, 158, 11, 0.07)';
            ctx.fillRect(left, y70, right - left, yBottom - y70);
        }

        // Target ceiling line (180)
        ctx.strokeStyle = 'rgba(20, 184, 166, 0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(left, y180);
        ctx.lineTo(right, y180);
        ctx.stroke();

        // Danger line (250)
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
        ctx.beginPath();
        ctx.moveTo(left, y250);
        ctx.lineTo(right, y250);
        ctx.stroke();

        ctx.restore();
    }
};

/**
 * Render Section 2: Timeline Chart (CGM + Heart Rate + 2h Forecast)
 */
function renderTimelineChart(timeline, prediction) {
    const canvas = document.getElementById('timeline-chart');
    if (!canvas) return;

    if (state.timelineChart) {
        state.timelineChart.destroy();
        state.timelineChart = null;
    }

    const labels = timeline.timestamps.map(formatTime);
    const glucoseData = [...timeline.glucose];
    const hrData = [...timeline.heart_rate];

    // Add forecast point 24 steps (2h) in future
    const lastLabel = labels[labels.length - 1] || '00:00';
    labels.push(`${lastLabel} (+2h)`);
    glucoseData.push(null); // Keep array aligned for past
    hrData.push(null);

    // Create forecast point dataset
    const forecastArray = new Array(labels.length).fill(null);
    forecastArray[forecastArray.length - 1] = prediction.predicted_glucose_2h;
    // Link line from current reading to predicted reading
    const forecastLineArray = new Array(labels.length).fill(null);
    forecastLineArray[forecastLineArray.length - 2] = timeline.glucose[timeline.glucose.length - 1];
    forecastLineArray[forecastLineArray.length - 1] = prediction.predicted_glucose_2h;

    const riskColor = RISK_COLORS[prediction.risk_level.toLowerCase()] || '#22c55e';

    const config = {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'CGM Glucose',
                    data: glucoseData,
                    borderColor: '#14b8a6',
                    backgroundColor: 'rgba(20, 184, 166, 0.1)',
                    borderWidth: 2.2,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    yAxisID: 'yGlucose',
                },
                {
                    label: 'Heart Rate',
                    data: hrData,
                    borderColor: '#3b82f6',
                    borderWidth: 1.2,
                    borderDash: [3, 3],
                    pointRadius: 0,
                    tension: 0.3,
                    yAxisID: 'yHR',
                },
                {
                    label: 'Forecast Trajectory',
                    data: forecastLineArray,
                    borderColor: riskColor,
                    borderWidth: 2,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    yAxisID: 'yGlucose',
                },
                {
                    label: '2-Hour Forecast Point',
                    data: forecastArray,
                    borderColor: riskColor,
                    backgroundColor: riskColor,
                    pointRadius: 7,
                    pointHoverRadius: 9,
                    showLine: false,
                    yAxisID: 'yGlucose',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: false, // Custom legend in HTML
                },
                tooltip: {
                    backgroundColor: '#111827',
                    titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1',
                    borderColor: '#374151',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return null;
                            if (context.dataset.yAxisID === 'yGlucose') {
                                return ` ${context.dataset.label}: ${context.parsed.y} mg/dL`;
                            }
                            return ` ${context.dataset.label}: ${context.parsed.y} bpm`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#94a3b8',
                        maxTicksLimit: 12,
                        maxRotation: 0,
                        font: { family: 'Inter', size: 11 },
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)',
                    }
                },
                yGlucose: {
                    type: 'linear',
                    position: 'left',
                    min: 40,
                    max: 380,
                    title: {
                        display: true,
                        text: 'Glucose (mg/dL)',
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 11, weight: '500' }
                    },
                    ticks: {
                        color: '#94a3b8',
                        stepSize: 50,
                        font: { family: 'JetBrains Mono', size: 10 }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                    }
                },
                yHR: {
                    type: 'linear',
                    position: 'right',
                    min: 40,
                    max: 160,
                    title: {
                        display: true,
                        text: 'Heart Rate (bpm)',
                        color: '#60a5fa',
                        font: { family: 'Inter', size: 11, weight: '500' }
                    },
                    ticks: {
                        color: '#60a5fa',
                        stepSize: 30,
                        font: { family: 'JetBrains Mono', size: 10 }
                    },
                    grid: {
                        drawOnChartArea: false,
                    }
                }
            }
        },
        plugins: [zoneBackgroundPlugin]
    };

    state.timelineChart = new Chart(canvas, config);
}

/**
 * Render Current Readings Bar
 */
function renderCurrentReadings(timeline) {
    const container = document.getElementById('current-readings');
    const n = timeline.glucose.length;
    if (n === 0) return;

    const curGlucose = timeline.glucose[n - 1];
    const curHR = timeline.heart_rate[n - 1];
    const curHRV = timeline.hrv_rmssd[n - 1];
    // Steps last 1 hour (last 12 readings)
    const recentSteps = timeline.steps.slice(-12);
    const steps1h = recentSteps.reduce((a, b) => a + b, 0);

    let gColor = '#22c55e';
    if (curGlucose > 250) gColor = '#ef4444';
    else if (curGlucose > 180) gColor = '#f59e0b';
    else if (curGlucose < 70) gColor = '#f59e0b';

    const hrColor = curHR > 100 ? '#f59e0b' : '#3b82f6';
    const hrvColor = curHRV < 25 ? '#f59e0b' : '#14b8a6';

    container.innerHTML = `
        <div class="vital-metric-card">
            <span class="vital-icon">🩸</span>
            <div class="vital-details">
                <span class="vital-label">Current CGM Glucose</span>
                <span class="vital-value" style="color:${gColor};">${curGlucose} <span class="vital-unit">mg/dL</span></span>
            </div>
        </div>
        <div class="vital-metric-card">
            <span class="vital-icon">❤️</span>
            <div class="vital-details">
                <span class="vital-label">Live Heart Rate</span>
                <span class="vital-value" style="color:${hrColor};">${curHR} <span class="vital-unit">bpm</span></span>
            </div>
        </div>
        <div class="vital-metric-card">
            <span class="vital-icon">🫀</span>
            <div class="vital-details">
                <span class="vital-label">HRV (RMSSD)</span>
                <span class="vital-value" style="color:${hrvColor};">${curHRV} <span class="vital-unit">ms</span></span>
            </div>
        </div>
        <div class="vital-metric-card">
            <span class="vital-icon">👟</span>
            <div class="vital-details">
                <span class="vital-label">Activity (Past Hour)</span>
                <span class="vital-value">${steps1h} <span class="vital-unit">steps</span></span>
            </div>
        </div>
    `;
}

/**
 * Render Section 3: Risk Gauge
 */
function renderRiskGauge(prediction) {
    const probEl = document.getElementById('risk-probability');
    const levelEl = document.getElementById('risk-level-text');
    const predValEl = document.getElementById('predicted-glucose-value');
    const gaugeFill = document.getElementById('gauge-fill');

    const prob = prediction.adverse_event_probability;
    const pct = Math.round(prob * 100);
    const riskLevel = prediction.risk_level.toLowerCase();
    const color = RISK_COLORS[riskLevel] || '#22c55e';

    if (probEl) probEl.textContent = `${pct}%`;
    if (predValEl) predValEl.textContent = prediction.predicted_glucose_2h.toFixed(1);

    if (levelEl) {
        levelEl.className = 'risk-level-display';
        levelEl.style.color = color;
        levelEl.textContent = prediction.risk_level.toUpperCase();
    }

    if (gaugeFill) {
        // Semi-circle arc length = pi * 80 ≈ 251.3
        const maxArc = 251.3;
        const targetOffset = maxArc * (1 - Math.min(1.0, Math.max(0.0, prob)));
        gaugeFill.style.stroke = color;
        gaugeFill.style.strokeDashoffset = targetOffset;
    }
}

/**
 * Render Section 4: SHAP Explainer
 */
function renderShapExplainer(prediction) {
    const container = document.getElementById('shap-container');
    if (!container) return;

    const features = prediction.shap_top_features || [];
    if (features.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted);font-size:12px;">No SHAP explanation features returned.</p>';
        return;
    }

    const maxAbsVal = Math.max(...features.map(f => Math.abs(f.contribution)), 0.01);

    let html = '';
    features.forEach(f => {
        const isStatic = f.stream === 'static';
        const streamClass = isStatic ? 'static' : 'dynamic';
        const arrow = f.contribution >= 0 ? '↑' : '↓';
        const sign = f.contribution >= 0 ? '+' : '';
        const pctWidth = Math.min(100, Math.max(4, (Math.abs(f.contribution) / maxAbsVal) * 100));
        const valClass = f.contribution >= 0 ? 'positive' : 'negative';

        html += `
            <div class="shap-bar-row">
                <div class="shap-feature-name" title="${f.feature} (Val: ${f.value})">
                    <span class="shap-stream-indicator" style="background:${isStatic ? '#3b82f6' : '#f97316'};"></span>
                    <span>${f.feature}</span>
                </div>
                <div class="shap-track">
                    <div class="shap-fill ${streamClass}" style="width:${pctWidth}%;"></div>
                </div>
                <div class="shap-value-text ${valClass}">
                    ${arrow} ${sign}${f.contribution.toFixed(1)}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

/**
 * Render Section 5: What-If Simulation Cards
 */
function renderSimulator(simulation) {
    const container = document.getElementById('simulator-container');
    if (!container) return;

    // Destroy existing scenario charts
    state.scenarioCharts.forEach(chart => {
        if (chart) chart.destroy();
    });
    state.scenarioCharts = [];

    const scenarios = simulation.scenarios || [];
    if (scenarios.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted);grid-column:1/-1;">No scenarios available.</p>';
        return;
    }

    let html = '';
    scenarios.forEach((sc, idx) => {
        const isBaseline = idx === 0;
        const isDanger = sc.enters_danger_zone;
        const borderClass = isDanger ? 'danger-border' : 'safe-border';
        const baselineClass = isBaseline ? 'scenario-baseline' : '';

        html += `
            <div class="scenario-card ${borderClass} ${baselineClass}">
                <div class="scenario-header">
                    <span class="scenario-title">${sc.name}</span>
                    ${getRiskBadgeHTML(sc.risk_level)}
                </div>
                <div class="mini-chart-container">
                    <canvas id="scenario-chart-${idx}"></canvas>
                </div>
                <div class="scenario-metrics">
                    <div class="scenario-metric">
                        <span class="scenario-metric-label">Peak Glucose</span>
                        <span class="scenario-metric-value" style="color:${sc.peak_glucose > 250 ? '#ef4444' : 'inherit'};">${sc.peak_glucose} mg/dL</span>
                    </div>
                    <div class="scenario-metric">
                        <span class="scenario-metric-label">Time-in-Range</span>
                        <span class="scenario-metric-value" style="color:${sc.time_in_range_pct >= 70 ? '#22c55e' : '#f59e0b'};">${sc.time_in_range_pct}%</span>
                    </div>
                </div>
                ${isDanger ? '<div class="scenario-danger-alert">⚠️ Exceeds Danger Threshold (>250)</div>' : '<div style="font-size:10px;color:var(--success);">✅ Safe Physiological Range</div>'}
            </div>
        `;
    });

    container.innerHTML = html;

    // Render mini trajectory charts for each scenario
    scenarios.forEach((sc, idx) => {
        const canvas = document.getElementById(`scenario-chart-${idx}`);
        if (!canvas) return;

        const times = sc.trajectory.map(p => `+${p.time_offset_min}m`);
        const values = sc.trajectory.map(p => p.glucose);

        const riskColor = RISK_COLORS[sc.risk_level.toLowerCase()] || '#22c55e';

        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: times,
                datasets: [
                    {
                        data: values,
                        borderColor: riskColor,
                        backgroundColor: `${riskColor}15`,
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0,
                        fill: true,
                    },
                    // Reference line at 250 (danger)
                    {
                        data: new Array(times.length).fill(250),
                        borderColor: 'rgba(239, 68, 68, 0.4)',
                        borderWidth: 1,
                        borderDash: [3, 3],
                        pointRadius: 0,
                        fill: false,
                    },
                    // Reference line at 180 (ceiling)
                    {
                        data: new Array(times.length).fill(180),
                        borderColor: 'rgba(245, 158, 11, 0.3)',
                        borderWidth: 1,
                        borderDash: [3, 3],
                        pointRadius: 0,
                        fill: false,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#111827',
                        padding: 6,
                        callbacks: {
                            label: (ctx) => ctx.datasetIndex === 0 ? ` ${ctx.parsed.y} mg/dL` : null
                        }
                    }
                },
                scales: {
                    x: {
                        display: false,
                    },
                    y: {
                        display: false,
                        min: Math.min(60, ...values) - 10,
                        max: Math.max(300, ...values) + 10,
                    }
                }
            }
        });

        state.scenarioCharts.push(chart);
    });
}

// 6. Main Orchestration Flow
async function loadPatient(patientId) {
    showLoading();
    try {
        state.currentPatientId = patientId;

        // Fetch all endpoints concurrently
        const [profile, timeline, prediction, simulation] = await Promise.all([
            loadPatientProfile(patientId),
            loadTimeline(patientId, 12),
            loadPrediction(patientId),
            loadSimulation(patientId),
        ]);

        state.patientProfile = profile;
        state.timeline = timeline;
        state.prediction = prediction;
        state.simulation = simulation;

        // Render all 5 sections
        renderPatientCard(profile);
        renderTimelineChart(timeline, prediction);
        renderCurrentReadings(timeline);
        renderRiskGauge(prediction);
        renderShapExplainer(prediction);
        renderSimulator(simulation);

    } catch (error) {
        console.error('Failed to load patient data:', error);
        alert(`Failed to load patient ${patientId}. Make sure the FastAPI backend is running on port 8000.`);
    } finally {
        hideLoading();
    }
}

// 7. Initialization
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const data = await loadPatientList();
        const selector = document.getElementById('patient-selector');
        selector.innerHTML = '';

        if (!data.patients || data.patients.length === 0) {
            throw new Error('No patients returned by API');
        }

        data.patients.forEach(id => {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `Patient ${id}`;
            selector.appendChild(option);
        });

        selector.addEventListener('change', (e) => {
            if (e.target.value) {
                loadPatient(e.target.value);
            }
        });

        // Auto-select first patient
        selector.value = data.patients[0];
        await loadPatient(data.patients[0]);

    } catch (error) {
        console.error('Dashboard initialization failed:', error);
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.innerHTML = `
                <div style="background:#111827;border:1px solid #ef4444;border-radius:12px;padding:24px;max-width:440px;text-align:center;">
                    <div style="font-size:36px;margin-bottom:12px;">⚠️</div>
                    <h2 style="color:#ef4444;font-size:18px;margin-bottom:8px;">Cannot Connect to Twin API</h2>
                    <p style="color:#94a3b8;font-size:13px;margin-bottom:16px;">
                        The dashboard requires the FastAPI server running on <code>http://localhost:8000</code>.
                    </p>
                    <div style="background:#1f2937;padding:10px;border-radius:6px;font-family:monospace;font-size:12px;color:#f1f5f9;margin-bottom:16px;">
                        python src/api/app.py
                    </div>
                    <button onclick="location.reload()" style="background:#14b8a6;color:#fff;border:none;padding:8px 18px;border-radius:6px;cursor:pointer;font-weight:600;">
                        Retry Connection
                    </button>
                </div>
            `;
        }
    }
});
