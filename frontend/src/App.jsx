import React, { useState, useEffect } from 'react';
import {
  Zap,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Layers,
  ChevronDown,
  ChevronUp,
  Activity,
  BookOpen,
  Info
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
  { id: 'Random Forest', name: 'Random Forest (Bagging Ensemble)' },
  { id: 'Gradient Boosting (GBM)', name: 'Gradient Boosting (GBM)' }
];

export default function App() {
  const [selectedBreaker, setSelectedBreaker] = useState('A');
  const [selectedModel, setSelectedModel] = useState('XGBoost (Fine-Tuned)');
  const [selectedScenario, setSelectedScenario] = useState('Normal Operating Condition');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showReferences, setShowReferences] = useState(false);
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
                  padding: '1.5rem',
                  marginBottom: '1.75rem',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.04)'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div
                    style={{
                      width: '48px',
                      height: '48px',
                      borderRadius: '50%',
                      backgroundColor: isMaint ? '#FEE2E2' : '#DCFCE7',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: isMaint ? '#DC2626' : '#16A34A'
                    }}
                  >
                    {isMaint ? <AlertTriangle size={28} /> : <CheckCircle2 size={28} />}
                  </div>
                  <div>
                    <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#64748B', textTransform: 'uppercase' }}>
                      Diagnostic Verdict
                    </span>
                    <h3 style={{ fontSize: '1.5rem', fontWeight: 800, color: isMaint ? '#DC2626' : '#16A34A', margin: '2px 0' }}>
                      {result.predicted_label}
                    </h3>
                    <p style={{ fontSize: '0.85rem', color: '#475569', maxWidth: '620px' }}>
                      {result.explanation}
                    </p>
                  </div>
                </div>

                <div style={{ textAlign: 'right', paddingLeft: '1.5rem', borderLeft: '1px solid #E2E8F0', minWidth: '150px' }}>
                  <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#64748B', textTransform: 'uppercase' }}>
                    Estimated Class Probability
                  </div>
                  <div style={{ fontSize: '2rem', fontWeight: 800, color: isMaint ? '#DC2626' : '#16A34A' }}>
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <span style={{ fontSize: '0.68rem', color: '#94A3B8', display: 'block', marginTop: '2px' }}>
                    Calibrated output (not guaranteed accuracy)
                  </span>
                </div>
              </div>
            ) : (
              <div
                style={{
                  backgroundColor: '#FFFFFF',
                  border: '1px dashed #CBD5E1',
                  borderRadius: '12px',
                  padding: '1.5rem',
                  marginBottom: '1.75rem',
                  textAlign: 'center',
                  color: '#64748B'
                }}
              >
                <p style={{ fontSize: '0.95rem', fontWeight: 600, color: '#334155' }}>
                  ⚡ Ready to assess circuit breaker health
                </p>
                <p style={{ fontSize: '0.82rem', marginTop: '4px' }}>
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
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>{features.SF6_density}</span>
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
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>{features.Ambient_cabinet_temperature}°C</span>
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
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>{features.Motor_run_time}s</span>
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
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>{features.Line_voltage?.toLocaleString()} V</span>
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
                    <span style={{ color: '#1E88E5', fontWeight: 700 }}>{features.Total_operation_count?.toLocaleString()}</span>
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

            {/* Optional Collapsible Advanced Details */}
            <div style={{ borderTop: '1px solid #E2E8F0', paddingTop: '1rem', marginBottom: '1.5rem' }}>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  background: 'none',
                  border: 'none',
                  color: '#64748B',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                {showAdvanced ? 'Hide Additional Technical Sensors' : 'Show Additional Technical Sensors (22 More)'}
              </button>

              {showAdvanced && (
                <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', fontSize: '0.8rem' }}>
                  <div className="app-card" style={{ padding: '0.75rem' }}>
                    <span style={{ color: '#64748B' }}>Open Latch Time:</span> <strong>{features.Open_latch_time}s</strong>
                  </div>
                  <div className="app-card" style={{ padding: '0.75rem' }}>
                    <span style={{ color: '#64748B' }}>Close Latch Time:</span> <strong>{features.Close_latch_time}s</strong>
                  </div>
                  <div className="app-card" style={{ padding: '0.75rem' }}>
                    <span style={{ color: '#64748B' }}>Fault Operations:</span> <strong>{features.Fault_operations_count}</strong>
                  </div>
                  <div className="app-card" style={{ padding: '0.75rem' }}>
                    <span style={{ color: '#64748B' }}>Contact Wear Life:</span> <strong>{features.Contact_life?.toLocaleString()}</strong>
                  </div>
                  <div className="app-card" style={{ padding: '0.75rem' }}>
                    <span style={{ color: '#64748B' }}>Arcing Time:</span> <strong>{features.Open_operation_arcing_time}s</strong>
                  </div>
                  <div className="app-card" style={{ padding: '0.75rem' }}>
                    <span style={{ color: '#64748B' }}>Motor I²t Stress:</span> <strong>{features.Motor_I2T} A²·s</strong>
                  </div>
                </div>
              )}
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
                    <p style={{ fontSize: '0.85rem', color: '#475569' }}>
                      <strong>{comparison.agreement_percentage}% Model Agreement</strong> ({comparison.vote_summary['Maintenance Required']} voted Maintenance, {comparison.vote_summary['No Maintenance Required']} voted Normal).
                      <span style={{ display: 'block', fontSize: '0.75rem', color: '#64748B', marginTop: '2px' }}>
                        Agreement reflects voting consensus among models; it is not a guarantee of operational accuracy.
                      </span>
                    </p>
                  </div>

                  <button onClick={handleCompare} className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
                    <RotateCcw size={14} /> Re-Evaluate
                  </button>
                </div>

                {/* 4 Models Comparison Grid (2 in each row) */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.25rem', marginBottom: '2rem' }}>
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
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                          <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#0F172A' }}>
                            {m.model_name}
                          </h4>
                          <span
                            style={{
                              display: 'inline-block',
                              padding: '0.2rem 0.55rem',
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

                        {/* Estimated Class Probability Bar */}
                        <div style={{ marginBottom: '0.85rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '4px' }}>
                            <span>Estimated Class Probability</span>
                            <span style={{ color: isModelMaint ? '#DC2626' : '#16A34A', fontWeight: 700 }}>
                              {(m.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div style={{ height: '6px', backgroundColor: '#F1F5F9', borderRadius: '3px', overflow: 'hidden' }}>
                            <div
                              style={{
                                height: '100%',
                                width: `${m.confidence * 100}%`,
                                backgroundColor: isModelMaint ? '#DC2626' : '#16A34A'
                              }}
                            />
                          </div>
                        </div>

                        {/* Test Evaluation Metrics (Held-out 750 Test Breakers) */}
                        <div style={{ backgroundColor: '#F8FAFC', borderRadius: '6px', padding: '0.65rem 0.75rem', fontSize: '0.75rem', border: '1px solid #E2E8F0' }}>
                          <div style={{ fontWeight: 700, color: '#334155', marginBottom: '4px' }}>
                            Held-Out Test Results (750 Samples):
                          </div>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '4px', color: '#475569' }}>
                            <div>Accuracy: <strong>{(m.test_accuracy * 100).toFixed(1)}%</strong></div>
                            <div>F1 Score (Pos=1): <strong>{(m.test_f1 * 100).toFixed(1)}%</strong></div>
                            <div>Precision: <strong>{(m.test_precision * 100).toFixed(1)}%</strong></div>
                            <div>Recall: <strong>{(m.test_recall * 100).toFixed(1)}%</strong></div>
                            <div style={{ gridColumn: 'span 2' }}>ROC-AUC: <strong>{m.test_roc_auc?.toFixed(4)}</strong></div>
                          </div>

                          {/* Compact Confusion Matrix */}
                          {cm.tn !== undefined && (
                            <div style={{ marginTop: '6px', paddingTop: '6px', borderTop: '1px dashed #CBD5E1', fontSize: '0.7rem', color: '#64748B' }}>
                              <span>Confusion Matrix: </span>
                              <strong style={{ color: '#16A34A' }}>TN: {cm.tn}</strong> • <strong style={{ color: '#D97706' }}>FP: {cm.fp}</strong> | <strong style={{ color: '#EF4444' }}>FN: {cm.fn}</strong> • <strong style={{ color: '#16A34A' }}>TP: {cm.tp}</strong>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '3rem', color: '#64748B' }}>
                <p>Click "Compare with Different Models" in the sidebar to run all models simultaneously.</p>
              </div>
            )}
          </div>
        )}

        {/* ------------------------------------------------------------- */}
        {/* VERIFIED REFERENCES & METHODOLOGY FOOTER                      */}
        {/* ------------------------------------------------------------- */}
        <div style={{ borderTop: '1px solid #E2E8F0', paddingTop: '1.25rem', marginTop: '1.5rem' }}>
          <button
            onClick={() => setShowReferences(!showReferences)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              background: 'none',
              border: 'none',
              color: '#475569',
              fontSize: '0.85rem',
              fontWeight: 700,
              cursor: 'pointer'
            }}
          >
            <BookOpen size={16} color="#1E88E5" />
            {showReferences ? 'Hide Verified Citations & References' : 'View Verified Citations & References'}
            {showReferences ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {showReferences && (
            <div style={{ marginTop: '0.75rem', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px', padding: '1rem', fontSize: '0.78rem', color: '#475569', lineHeight: 1.5 }}>
              <p style={{ fontWeight: 700, color: '#0F172A', marginBottom: '0.4rem' }}>
                Verified Official References & Data Provenance:
              </p>
              <ul style={{ paddingLeft: '1.25rem', margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <li>
                  <strong>Dataset & Substation Maintenance Benchmark:</strong> Ghosh S, Mittal G (2026). <em>"Federated learning for critical electrical infrastructure - handling data heterogeneity for predictive maintenance of substation equipment"</em>, Frontiers in Artificial Intelligence 8:1697175. GitHub repository: <a href="https://github.com/sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance" target="_blank" rel="noreferrer" style={{ color: '#1E88E5' }}>sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance</a> (Apache 2.0 License). <em>Note: As stated by the authors, this dataset is semi-synthetically derived for research and educational purposes based on utility substation operational monitoring guidelines.</em>
                </li>
                <li>
                  <strong>XGBoost Algorithm & Optimization:</strong> Chen T, Guestrin C (2016). <em>"XGBoost: A Scalable Tree Boosting System"</em>, Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), pp. 785–794. Official documentation: <a href="https://xgboost.readthedocs.io/" target="_blank" rel="noreferrer" style={{ color: '#1E88E5' }}>xgboost.readthedocs.io</a>.
                </li>
                <li>
                  <strong>Random Forest:</strong> Breiman L (2001). <em>"Random Forests"</em>, Machine Learning, 45(1), pp. 5–32. Scikit-learn documentation: <a href="https://scikit-learn.org/stable/modules/ensemble.html" target="_blank" rel="noreferrer" style={{ color: '#1E88E5' }}>scikit-learn.org/stable/modules/ensemble.html</a>.
                </li>
                <li>
                  <strong>Gradient Tree Boosting:</strong> Friedman JH (2001). <em>"Greedy Function Approximation: A Gradient Boosting Machine"</em>, The Annals of Statistics, 29(5), pp. 1189–1232.
                </li>
              </ul>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
