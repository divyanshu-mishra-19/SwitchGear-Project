import React, { useState, useEffect } from 'react';
import {
  Zap,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Layers,
  Activity
} from 'lucide-react';

const API_BASE = '/api';

const BREAKER_TYPES = [
  { id: 'A', name: 'Type A — SF6 Dead-Tank (145 kV)', desc: '145 kV Utility Grid Substation' },
  { id: 'B', name: 'Type B — SF6 Live-Tank (145 kV)', desc: '145 kV High-Voltage Transmission' },
  { id: 'C', name: 'Type C — Gas-Insulated GIS (126 kV)', desc: '126 kV Compact Substation' },
  { id: 'D', name: 'Type D — High-Voltage Outdoor (145 kV)', desc: '145 kV Heavy Duty Substation' },
  { id: 'E', name: 'Type E — Industrial Feeder (115 kV)', desc: '115 kV Distribution Feeder' }
];

const MODELS = [
  { id: 'XGBoost (Fine-Tuned)', name: 'XGBoost (Fine-Tuned) — Primary' },
  { id: 'XGBoost (Baseline)', name: 'XGBoost (Baseline / Default)' },
  { id: 'Random Forest', name: 'Random Forest' },
  { id: 'Gradient Boosting (GBM)', name: 'Gradient Boosting (GBM)' }
];

export default function App() {
  const [selectedBreaker, setSelectedBreaker] = useState('A');
  const [selectedModel, setSelectedModel] = useState('XGBoost (Fine-Tuned)');
  const [selectedScenario, setSelectedScenario] = useState('Normal Operating Condition');
  const [viewMode, setViewMode] = useState('single'); // 'single' | 'compare'

  const [scenarios, setScenarios] = useState({});
  const [loading, setLoading] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [backendReady, setBackendReady] = useState(false);

  // Core features state initialized with nominal operating telemetry
  const [features, setFeatures] = useState({
    Product_variant: 'A',
    Breaker_status: 'Closed',
    SF6_density: 6.05,
    SF6_temperature: 20.2,
    SF6_dew_point: 19.8,
    Motor_run_time: 5.5,
    Motor_runs_per_day: 1,
    Motor_I2T: 7.2,
    Heater_status: 'Off',
    Line_voltage: 120000,
    Ambient_cabinet_temperature: 58.5,
    Days_since_last_operation: 30,
    Contact_life: 33000,
    Last_trip_coil_energized: 'TC1',
    Clearing_time: 0.045,
    Open_latch_time: 0.095,
    Close_latch_time: 0.098,
    Opening_travel_time: 0.078,
    Closing_travel_time: 0.079,
    Total_operation_count: 4200,
    Non_fault_operation_count: 3650,
    Fault_operations_count: 550,
    Accumulated_motor_run_time: 420.0,
    Heater_current: 0.0,
    Open_average_velocity: 3.1,
    Close_average_velocity: 3.1,
    Restrike_occurrence: 0.0,
    Open_operation_arcing_time: 0.024
  });

  const [result, setResult] = useState(null);
  const [comparison, setComparison] = useState(null);

  // Load scenarios from backend on mount
  useEffect(() => {
    fetchScenarios();
  }, []);

  const fetchScenarios = async () => {
    try {
      const res = await fetch(`${API_BASE}/scenarios`);
      if (res.ok) {
        const data = await res.json();
        setScenarios(data);
        setBackendReady(true);
        if (data['Normal Operating Condition']) {
          applyScenario('Normal Operating Condition', data);
        }
      }
    } catch (e) {
      console.warn('Backend not ready yet, retrying in 1s...', e);
      setTimeout(fetchScenarios, 1000);
    }
  };

  const applyScenario = (scenName, scenData = scenarios) => {
    setSelectedScenario(scenName);
    if (scenData && scenData[scenName]) {
      const scen = scenData[scenName];
      const newF = { ...scen.features };
      if (newF.Product_variant) {
        setSelectedBreaker(newF.Product_variant);
      }
      setFeatures(newF);
      setResult(null);
      setComparison(null);
    }
  };

  const handleBreakerChange = (typeId) => {
    setSelectedBreaker(typeId);
    setFeatures((prev) => ({
      ...prev,
      Product_variant: typeId
    }));
    setResult(null);
    setComparison(null);
  };

  const handleSliderChange = (field, val) => {
    setFeatures((prev) => ({
      ...prev,
      [field]: val
    }));
  };

  // Run single prediction
  const handlePredict = async () => {
    setLoading(true);
    setViewMode('single');
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: features,
          model_name: selectedModel
        })
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
      } else {
        alert('Prediction request failed. Check server status.');
      }
    } catch (e) {
      alert('Cannot connect to backend: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Run multi-model comparison
  const handleCompare = async () => {
    setComparing(true);
    setViewMode('compare');
    try {
      const res = await fetch(`${API_BASE}/compare-models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(features)
      });
      if (res.ok) {
        const data = await res.json();
        setComparison(data);
      } else {
        alert('Comparison request failed.');
      }
    } catch (e) {
      alert('Error connecting to backend: ' + e.message);
    } finally {
      setComparing(false);
    }
  };

  const isMaint = result && result.predicted_class === 1;

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F8FAFC' }}>
      {/* ------------------------------------------------------------- */}
      {/* SIDEBAR: SIMPLE, ESSENTIAL CONTROLS ONLY                       */}
      {/* ------------------------------------------------------------- */}
      <aside
        style={{
          width: '320px',
          minWidth: '320px',
          backgroundColor: '#FFFFFF',
          borderRight: '1px solid #E2E8F0',
          display: 'flex',
          flexDirection: 'column',
          padding: '1.5rem',
          gap: '1.25rem',
          boxShadow: '1px 0 3px rgba(0,0,0,0.02)'
        }}
      >
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', paddingBottom: '1rem', borderBottom: '1px solid #F1F5F9' }}>
          <div
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '8px',
              backgroundColor: '#1E88E5',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white'
            }}
          >
            <Zap size={20} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0F172A', letterSpacing: '-0.3px' }}>
              CircuitGuard AI
            </h1>
            <span style={{ fontSize: '0.75rem', color: '#64748B' }}>
              Predictive Maintenance
            </span>
          </div>
        </div>

        {/* 1. Circuit Breaker Type Dropdown */}
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 700, color: '#334155', marginBottom: '0.4rem', textTransform: 'uppercase' }}>
            Circuit Breaker Type
          </label>
          <select
            className="input-field"
            value={selectedBreaker}
            onChange={(e) => handleBreakerChange(e.target.value)}
            style={{ fontWeight: 600, cursor: 'pointer' }}
          >
            {BREAKER_TYPES.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </select>
          <span style={{ fontSize: '0.72rem', color: '#64748B', display: 'block', marginTop: '4px' }}>
            {BREAKER_TYPES.find((b) => b.id === selectedBreaker)?.desc}
          </span>
        </div>

        {/* 2. Demonstration Scenario Dropdown */}
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 700, color: '#334155', marginBottom: '0.4rem', textTransform: 'uppercase' }}>
            Preset Condition
          </label>
          <select
            className="input-field"
            value={selectedScenario}
            onChange={(e) => applyScenario(e.target.value)}
            style={{ fontWeight: 500, cursor: 'pointer' }}
          >
            <option value="Custom Manual Input">Custom Manual Input</option>
            {Object.keys(scenarios).map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        {/* 3. ML Model Dropdown */}
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 700, color: '#334155', marginBottom: '0.4rem', textTransform: 'uppercase' }}>
            Model
          </label>
          <select
            className="input-field"
            value={selectedModel}
            onChange={(e) => {
              setSelectedModel(e.target.value);
              setResult(null);
            }}
            style={{ fontWeight: 500, cursor: 'pointer' }}
          >
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>

        {/* 4. Action Buttons */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', marginTop: '0.5rem' }}>
          <button
            onClick={handlePredict}
            disabled={loading}
            className="btn-primary"
            style={{ padding: '0.8rem', fontSize: '0.9rem' }}
          >
            <Zap size={18} />
            {loading ? 'Evaluating...' : 'Predict Maintenance'}
          </button>

          <button
            onClick={handleCompare}
            disabled={comparing}
            className="btn-secondary"
            style={{ padding: '0.75rem', borderColor: '#1E88E5', color: '#1E88E5' }}
          >
            <Layers size={18} />
            {comparing ? 'Comparing...' : 'Compare with Different Models'}
          </button>

          <button
            onClick={() => applyScenario(selectedScenario)}
            className="btn-secondary"
            style={{ fontSize: '0.8rem', padding: '0.5rem' }}
          >
            <RotateCcw size={14} /> Reset Values
          </button>
        </div>

        {/* Status */}
        <div style={{ marginTop: 'auto', borderTop: '1px solid #F1F5F9', paddingTop: '1rem', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: backendReady ? '#16A34A' : '#D97706',
              display: 'inline-block'
            }}
          />
          <span style={{ color: backendReady ? '#16A34A' : '#D97706', fontWeight: 600 }}>
            {backendReady ? 'Inference Core Active' : 'Connecting to Core...'}
          </span>
        </div>
      </aside>

      {/* ------------------------------------------------------------- */}
      {/* MAIN CONTENT: SIMPLE, INTUITIVE DISPLAY                       */}
      {/* ------------------------------------------------------------- */}
      <main style={{ flex: 1, padding: '2rem', maxWidth: '1100px', margin: '0 auto', overflowY: 'auto' }}>
        {/* Top Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#0F172A' }}>
              {viewMode === 'compare' ? 'Multi-Model Benchmark & Comparison' : 'Circuit Breaker Assessment'}
            </h2>
            <p style={{ fontSize: '0.85rem', color: '#64748B', marginTop: '2px' }}>
              {BREAKER_TYPES.find((b) => b.id === selectedBreaker)?.name} • Operating under {selectedModel}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              onClick={() => setViewMode('single')}
              style={{
                padding: '0.45rem 0.9rem',
                borderRadius: '6px',
                border: '1px solid #E2E8F0',
                backgroundColor: viewMode === 'single' ? '#1E88E5' : '#FFFFFF',
                color: viewMode === 'single' ? '#FFFFFF' : '#475569',
                fontWeight: 600,
                fontSize: '0.8rem',
                cursor: 'pointer'
              }}
            >
              Dashboard
            </button>
            <button
              onClick={() => {
                setViewMode('compare');
                if (!comparison) handleCompare();
              }}
              style={{
                padding: '0.45rem 0.9rem',
                borderRadius: '6px',
                border: '1px solid #E2E8F0',
                backgroundColor: viewMode === 'compare' ? '#1E88E5' : '#FFFFFF',
                color: viewMode === 'compare' ? '#FFFFFF' : '#475569',
                fontWeight: 600,
                fontSize: '0.8rem',
                cursor: 'pointer'
              }}
            >
              Compare Models
            </button>
          </div>
        </div>

        {/* ------------------------------------------------------------- */}
        {/* VIEW 1: SINGLE PREDICTION VIEW                                */}
        {/* ------------------------------------------------------------- */}
        {viewMode === 'single' && (
          <div>
            {/* 1. Prediction Result Card */}
            {result ? (
              <div
                style={{
                  backgroundColor: isMaint ? '#FEF2F2' : '#F0FDF4',
                  border: `2px solid ${isMaint ? '#EF4444' : '#22C55E'}`,
                  borderRadius: '12px',
                  padding: '1.25rem 1.5rem',
                  marginBottom: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1.25rem',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.04)'
                }}
              >
                <div
                  style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '50%',
                    backgroundColor: isMaint ? '#FEE2E2' : '#DCFCE7',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: isMaint ? '#DC2626' : '#16A34A',
                    flexShrink: 0
                  }}
                >
                  {isMaint ? <AlertTriangle size={28} /> : <CheckCircle2 size={28} />}
                </div>
                <div>
                  <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748B', textTransform: 'uppercase' }}>
                    Diagnostic Verdict
                  </span>
                  <h3 style={{ fontSize: '1.4rem', fontWeight: 800, color: isMaint ? '#DC2626' : '#16A34A', margin: '2px 0 0 0' }}>
                    {result.predicted_label}
                  </h3>
                </div>
              </div>
            ) : (
              <div
                style={{
                  backgroundColor: '#FFFFFF',
                  border: '1px dashed #CBD5E1',
                  borderRadius: '12px',
                  padding: '1.5rem',
                  marginBottom: '1.5rem',
                  textAlign: 'center',
                  color: '#64748B'
                }}
              >
                <p style={{ fontSize: '0.95rem', fontWeight: 600, color: '#334155', margin: 0 }}>
                  ⚡ Ready to assess circuit breaker health
                </p>
                <p style={{ fontSize: '0.82rem', marginTop: '4px', marginBottom: 0 }}>
                  Adjust the operating sliders below and click <strong>"Predict Maintenance"</strong> or choose a preset condition.
                </p>
              </div>
            )}

            {/* 2. Key Operating Sliders (Clean & Simple) */}
            <div className="app-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#0F172A', marginBottom: '1.25rem' }}>
                Primary Operating Readings
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
                {/* SF6 Density */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '6px' }}>
                    <span>SF6 Gas Density (kg/m³)</span>
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>
                      {Number(features.SF6_density).toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="5.0"
                    max="7.0"
                    step="0.01"
                    value={features.SF6_density}
                    onChange={(e) => handleSliderChange('SF6_density', parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#94A3B8' }}>
                    <span>5.0 (Low Density Alert)</span>
                    <span>Norm: ~6.0</span>
                    <span>7.0 (Nominal)</span>
                  </div>
                </div>

                {/* Cabinet Temperature */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '6px' }}>
                    <span>Ambient Cabinet Temperature (°C)</span>
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>
                      {Number(features.Ambient_cabinet_temperature).toFixed(1)}°C
                    </span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="120"
                    step="1"
                    value={features.Ambient_cabinet_temperature}
                    onChange={(e) => handleSliderChange('Ambient_cabinet_temperature', parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#94A3B8' }}>
                    <span>20°C (Normal)</span>
                    <span>65°C</span>
                    <span>120°C (Overheat)</span>
                  </div>
                </div>

                {/* Charging Motor Run Time */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '6px' }}>
                    <span>Spring Motor Run Time (seconds)</span>
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>
                      {Number(features.Motor_run_time).toFixed(1)}s
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1.0"
                    max="15.0"
                    step="0.5"
                    value={features.Motor_run_time}
                    onChange={(e) => handleSliderChange('Motor_run_time', parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#94A3B8' }}>
                    <span>1.0s (Fast/Healthy)</span>
                    <span>6.0s</span>
                    <span>15.0s (Sluggish/Wear)</span>
                  </div>
                </div>

                {/* Grid Line Voltage */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '6px' }}>
                    <span>Grid Line Voltage (V)</span>
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>
                      {Math.round(Number(features.Line_voltage)).toLocaleString()} V
                    </span>
                  </div>
                  <input
                    type="range"
                    min="72000"
                    max="150000"
                    step="2000"
                    value={features.Line_voltage}
                    onChange={(e) => handleSliderChange('Line_voltage', parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#94A3B8' }}>
                    <span>72 kV</span>
                    <span>120 kV Nominal</span>
                    <span>150 kV</span>
                  </div>
                </div>

                {/* Total Operation Count */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '6px' }}>
                    <span>Total Operations Count (cycles)</span>
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>
                      {Math.round(Number(features.Total_operation_count)).toLocaleString()}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1000"
                    max="6500"
                    step="100"
                    value={features.Total_operation_count}
                    onChange={(e) => handleSliderChange('Total_operation_count', parseInt(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#94A3B8' }}>
                    <span>1,000 (New)</span>
                    <span>4,000</span>
                    <span>6,500 (End of Life)</span>
                  </div>
                </div>

                {/* Breaker Contact Status */}
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '8px' }}>
                    Primary Breaker Status
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      onClick={() => handleSliderChange('Breaker_status', 'Closed')}
                      style={{
                        flex: 1,
                        padding: '0.5rem',
                        borderRadius: '6px',
                        border: '1px solid #CBD5E1',
                        backgroundColor: features.Breaker_status === 'Closed' ? '#E3F2FD' : '#FFFFFF',
                        color: features.Breaker_status === 'Closed' ? '#1E88E5' : '#475569',
                        fontWeight: 600,
                        cursor: 'pointer'
                      }}
                    >
                      Closed (In-Service)
                    </button>
                    <button
                      onClick={() => handleSliderChange('Breaker_status', 'Open')}
                      style={{
                        flex: 1,
                        padding: '0.5rem',
                        borderRadius: '6px',
                        border: '1px solid #CBD5E1',
                        backgroundColor: features.Breaker_status === 'Open' ? '#E3F2FD' : '#FFFFFF',
                        color: features.Breaker_status === 'Open' ? '#1E88E5' : '#475569',
                        fontWeight: 600,
                        cursor: 'pointer'
                      }}
                    >
                      Open (Isolated)
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ------------------------------------------------------------- */}
        {/* VIEW 2: MULTI-MODEL COMPARISON VIEW                           */}
        {/* ------------------------------------------------------------- */}
        {viewMode === 'compare' && (
          <div>
            {comparing ? (
              <div style={{ textAlign: 'center', padding: '3rem' }}>
                <Activity className="animate-spin" size={32} color="#1E88E5" style={{ margin: '0 auto 1rem' }} />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Evaluating Machine Learning Models...</h3>
              </div>
            ) : comparison ? (
              <div>
                {/* Consensus Banner */}
                <div
                  style={{
                    padding: '1.25rem 1.5rem',
                    borderRadius: '12px',
                    backgroundColor: comparison.consensus_label === 'Maintenance Required' ? '#FEF2F2' : '#F0FDF4',
                    border: `1px solid ${comparison.consensus_label === 'Maintenance Required' ? '#FECACA' : '#BBF7D0'}`,
                    marginBottom: '1.5rem',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}
                >
                  <div>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748B', textTransform: 'uppercase' }}>
                      Ensemble Voting Summary
                    </span>
                    <h3 style={{ fontSize: '1.4rem', fontWeight: 800, color: comparison.consensus_label === 'Maintenance Required' ? '#DC2626' : '#16A34A', margin: '2px 0' }}>
                      {comparison.consensus_label}
                    </h3>
                    <p style={{ fontSize: '0.85rem', color: '#475569', margin: 0 }}>
                      <strong>{comparison.agreement_percentage}% Model Agreement</strong> ({comparison.vote_summary['Maintenance Required']} Maintenance, {comparison.vote_summary['No Maintenance Required']} Normal)
                    </p>
                  </div>

                  <button onClick={handleCompare} className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
                    <RotateCcw size={14} /> Re-Evaluate
                  </button>
                </div>

                {/* 4 Models Comparison Grid (2 in each row) */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.25rem', marginBottom: '1.75rem' }}>
                  {comparison.comparisons.map((m, idx) => {
                    const isModelMaint = m.predicted_class === 1;
                    const cm = m.confusion_matrix || {};
                    return (
                      <div
                        key={idx}
                        className="app-card"
                        style={{
                          padding: '1.25rem',
                          borderTop: `4px solid ${isModelMaint ? '#DC2626' : '#16A34A'}`
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                          <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#0F172A', margin: 0 }}>
                            {m.model_name}
                          </h4>
                          <span
                            style={{
                              display: 'inline-block',
                              padding: '0.2rem 0.6rem',
                              borderRadius: '4px',
                              fontSize: '0.75rem',
                              fontWeight: 700,
                              backgroundColor: isModelMaint ? '#FEE2E2' : '#DCFCE7',
                              color: isModelMaint ? '#B91C1C' : '#15803D'
                            }}
                          >
                            {m.predicted_label}
                          </span>
                        </div>

                        {/* Test Evaluation Metrics (Held-out 750 Test Breakers) */}
                        <div style={{ backgroundColor: '#F8FAFC', borderRadius: '6px', padding: '0.75rem', fontSize: '0.75rem', border: '1px solid #E2E8F0' }}>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px', color: '#475569', marginBottom: '0.65rem' }}>
                            <div>Accuracy: <strong>{(m.test_accuracy * 100).toFixed(1)}%</strong></div>
                            <div>F1 Score: <strong>{(m.test_f1 * 100).toFixed(1)}%</strong></div>
                            <div>Precision: <strong>{(m.test_precision * 100).toFixed(1)}%</strong></div>
                            <div>Recall: <strong>{(m.test_recall * 100).toFixed(1)}%</strong></div>
                            <div style={{ gridColumn: 'span 2' }}>ROC-AUC: <strong>{m.test_roc_auc?.toFixed(4)}</strong></div>
                          </div>

                          {/* Confusion Matrix Formatted as 2x2 Table */}
                          {cm.tn !== undefined && (
                            <div style={{ paddingTop: '0.65rem', borderTop: '1px solid #E2E8F0' }}>
                              <table
                                style={{
                                  width: '100%',
                                  borderCollapse: 'collapse',
                                  fontSize: '0.72rem',
                                  textAlign: 'center',
                                  border: '1px solid #E2E8F0',
                                  borderRadius: '6px',
                                  overflow: 'hidden'
                                }}
                              >
                                <thead>
                                  <tr style={{ backgroundColor: '#F1F5F9', color: '#475569', fontSize: '0.68rem' }}>
                                    <th style={{ padding: '4px 6px', textAlign: 'left', border: '1px solid #E2E8F0', fontWeight: 600 }}>Actual \ Predicted</th>
                                    <th style={{ padding: '4px 6px', border: '1px solid #E2E8F0', fontWeight: 700, color: '#16A34A' }}>Pred: Normal (0)</th>
                                    <th style={{ padding: '4px 6px', border: '1px solid #E2E8F0', fontWeight: 700, color: '#DC2626' }}>Pred: Maint (1)</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  <tr>
                                    <td style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, backgroundColor: '#F8FAFC', border: '1px solid #E2E8F0', color: '#334155' }}>
                                      Actual: Normal (0)
                                    </td>
                                    <td style={{ padding: '4px 6px', fontWeight: 700, color: '#15803D', backgroundColor: '#F0FDF4', border: '1px solid #E2E8F0' }}>
                                      TN: {cm.tn}
                                    </td>
                                    <td style={{ padding: '4px 6px', fontWeight: 700, color: '#D97706', backgroundColor: '#FFFBEB', border: '1px solid #E2E8F0' }}>
                                      FP: {cm.fp}
                                    </td>
                                  </tr>
                                  <tr>
                                    <td style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, backgroundColor: '#F8FAFC', border: '1px solid #E2E8F0', color: '#334155' }}>
                                      Actual: Maint (1)
                                    </td>
                                    <td style={{ padding: '4px 6px', fontWeight: 700, color: '#DC2626', backgroundColor: '#FEF2F2', border: '1px solid #E2E8F0' }}>
                                      FN: {cm.fn}
                                    </td>
                                    <td style={{ padding: '4px 6px', fontWeight: 700, color: '#15803D', backgroundColor: '#F0FDF4', border: '1px solid #E2E8F0' }}>
                                      TP: {cm.tp}
                                    </td>
                                  </tr>
                                </tbody>
                              </table>

                              {/* Aligned Legend */}
                              <div
                                style={{
                                  display: 'grid',
                                  gridTemplateColumns: '1fr 1fr',
                                  gap: '3px 12px',
                                  fontSize: '0.68rem',
                                  marginTop: '6px',
                                  padding: '4px 8px',
                                  backgroundColor: '#F8FAFC',
                                  borderRadius: '4px',
                                  border: '1px solid #E2E8F0',
                                  color: '#475569'
                                }}
                              >
                                <div>
                                  <strong style={{ color: '#15803D' }}>TN:</strong> True Normal ({cm.tn})
                                </div>
                                <div>
                                  <strong style={{ color: '#D97706' }}>FP:</strong> False Alarm ({cm.fp})
                                </div>
                                <div>
                                  <strong style={{ color: '#DC2626' }}>FN:</strong> Missed Fault ({cm.fn})
                                </div>
                                <div>
                                  <strong style={{ color: '#15803D' }}>TP:</strong> True Maint ({cm.tp})
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Minimal Model Comparison Table */}
                <div className="app-card" style={{ padding: '1.25rem 1.5rem', marginBottom: '2rem' }}>
                  <div style={{ marginBottom: '0.85rem' }}>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#0F172A', margin: 0 }}>
                      Model Comparison Table
                    </h3>
                  </div>

                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'center' }}>
                      <thead>
                        <tr style={{ backgroundColor: '#F8FAFC', borderBottom: '2px solid #CBD5E1', color: '#334155', fontWeight: 700, fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.025em' }}>
                          <th style={{ padding: '0.65rem 0.75rem', textAlign: 'left' }}>Model</th>
                          <th style={{ padding: '0.65rem 0.5rem' }}>Accuracy</th>
                          <th style={{ padding: '0.65rem 0.5rem' }}>Precision</th>
                          <th style={{ padding: '0.65rem 0.5rem' }}>Recall</th>
                          <th style={{ padding: '0.65rem 0.5rem' }}>F1-Score</th>
                          <th style={{ padding: '0.65rem 0.5rem' }}>ROC-AUC</th>
                          <th style={{ padding: '0.65rem 0.5rem' }}>False Alarms (FP)</th>
                          <th style={{ padding: '0.65rem 0.5rem' }}>Missed Faults (FN)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparison.comparisons.map((m, idx) => {
                          const isPrimary = m.model_name.includes('Fine-Tuned');
                          const cm = m.confusion_matrix || {};

                          return (
                            <tr
                              key={idx}
                              style={{
                                borderBottom: '1px solid #E2E8F0',
                                backgroundColor: isPrimary ? '#F0F9FF' : (idx % 2 === 0 ? '#FFFFFF' : '#FAFAFA')
                              }}
                            >
                              <td style={{ padding: '0.65rem 0.75rem', textAlign: 'left', fontWeight: 700, color: isPrimary ? '#0369A1' : '#0F172A' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                  <span>{m.model_name}</span>
                                  {isPrimary && (
                                    <span style={{ fontSize: '0.65rem', padding: '0.12rem 0.35rem', backgroundColor: '#BAE6FD', color: '#0369A1', borderRadius: '4px', fontWeight: 800 }}>
                                      PRIMARY
                                    </span>
                                  )}
                                </div>
                              </td>
                              <td style={{ padding: '0.65rem 0.5rem', fontWeight: 700, color: '#0F172A' }}>
                                {(m.test_accuracy * 100).toFixed(1)}%
                              </td>
                              <td style={{ padding: '0.65rem 0.5rem', color: m.test_precision > 0.95 ? '#15803D' : '#334155', fontWeight: m.test_precision > 0.95 ? 700 : 500 }}>
                                {(m.test_precision * 100).toFixed(1)}%
                              </td>
                              <td style={{ padding: '0.65rem 0.5rem', color: m.test_recall > 0.85 ? '#15803D' : '#334155', fontWeight: m.test_recall > 0.85 ? 700 : 500 }}>
                                {(m.test_recall * 100).toFixed(1)}%
                              </td>
                              <td style={{ padding: '0.65rem 0.5rem', fontWeight: 700, color: '#0F172A' }}>
                                {(m.test_f1 * 100).toFixed(1)}%
                              </td>
                              <td style={{ padding: '0.65rem 0.5rem', fontWeight: 700, color: isPrimary ? '#0284C7' : '#334155' }}>
                                {m.test_roc_auc?.toFixed(4)}
                              </td>
                              <td style={{ padding: '0.65rem 0.5rem', color: cm.fp <= 7 ? '#15803D' : (cm.fp >= 20 ? '#DC2626' : '#D97706'), fontWeight: 700 }}>
                                {cm.fp ?? '—'}
                              </td>
                              <td style={{ padding: '0.65rem 0.5rem', color: cm.fn <= 45 ? '#15803D' : '#DC2626', fontWeight: 700 }}>
                                {cm.fn ?? '—'}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '3rem', color: '#64748B' }}>
                <p>Click "Compare with Different Models" in the sidebar to run all models simultaneously.</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
