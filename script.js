const API = 'http://127.0.0.1:5000/api';
let yieldData = null; // stored from yield prediction

// ============================
// PAGE NAVIGATION
// ============================
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  window.scrollTo(0, 0);
  if (name === 'price-from-yield') loadPriceFromYield();
}

// ============================
// LOAD META
// ============================
async function loadMeta() {
  try {
    const res = await fetch(API + '/meta');
    const meta = await res.json();

    // Yield dropdowns
    populateSelect('yield-crop', meta.yield_crops);
    populateSelect('yield-state', meta.yield_states);

    // Price dropdowns
    populateSelect('price-crop', meta.price_crops);
    populateSelect('price-state', meta.price_states);
  } catch (e) {
    console.error('Failed to load meta:', e);
    // Fallback static data
    populateSelect('yield-crop', ['Rice', 'Wheat', 'Maize', 'Bajra', 'Jowar', 'Groundnut', 'Urad', 'Moong']);
    populateSelect('yield-state', ['Andhra Pradesh', 'Bihar', 'Gujarat', 'Haryana', 'Karnataka', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal']);
    populateSelect('price-crop', ['Rice', 'Jowar', 'Bajra', 'Groundnut', 'Urad', 'Moong', 'Maize']);
    populateSelect('price-state', ['Andhra Pradesh', 'Assam', 'Bihar', 'Gujarat', 'Haryana', 'Karnataka', 'Madhya Pradesh', 'Maharashtra', 'Punjab', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal']);
  }
}

function populateSelect(id, items) {
  const sel = document.getElementById(id);
  sel.innerHTML = items.map(i => `<option value="${i}">${i}</option>`).join('');
}

// ============================
// PREDICT YIELD
// ============================
async function predictYield() {
  const crop = document.getElementById('yield-crop').value;
  const state = document.getElementById('yield-state').value;
  const season = document.getElementById('yield-season').value;
  const area = document.getElementById('yield-area').value;

  if (!crop || !state || !area) {
    document.getElementById('yield-error').innerHTML = '<div class="error-msg">⚠ Please fill all fields.</div>';
    return;
  }
  document.getElementById('yield-error').innerHTML = '';

  const btn = document.getElementById('yield-btn');
  btn.disabled = true; btn.textContent = 'Predicting...';

  const resultEl = document.getElementById('yield-result');
  resultEl.style.display = 'block';
  resultEl.innerHTML = `<div class="loader-wrap"><div class="spinner"></div><div class="loader-text">Fetching weather & running model…</div></div>`;
  document.getElementById('yield-price-section').style.display = 'none';

  try {
    const res = await fetch(API + '/predict_yield', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ crop, state, season, area: parseFloat(area) })
    });
    const data = await res.json();

    yieldData = { crop, state, season, area, yield_value: data.predicted_yield, weather: data.weather };

    // Update steps
    document.getElementById('ystep1').className = 'step done';
    document.getElementById('ystep2').className = 'step active';

    const w = data.weather;
    resultEl.innerHTML = `
      <div class="result-card">
        <div class="result-label">Predicted Crop Yield</div>
        <div class="result-value">${data.predicted_yield.toLocaleString('en-IN', {maximumFractionDigits:2})}</div>
        <div class="result-unit">kg per hectare</div>

        <div class="result-label" style="margin-bottom:10px">Live Weather Conditions</div>
        <div class="weather-row">
          <div class="weather-chip"><span class="icon">🌡</span>${w.temperature}°C</div>
          <div class="weather-chip"><span class="icon">💧</span>${w.humidity}% Humidity</div>
          <div class="weather-chip"><span class="icon">🌧</span>${w.rainfall.toFixed(1)} mm Rainfall</div>
        </div>

        <div class="result-label" style="margin-bottom:10px">Input Summary</div>
        <div class="info-grid">
          <div class="info-chip"><div class="ic-label">Crop</div><div class="ic-val">${crop}</div></div>
          <div class="info-chip"><div class="ic-label">State</div><div class="ic-val">${state}</div></div>
          <div class="info-chip"><div class="ic-label">Season</div><div class="ic-val">${season}</div></div>
          <div class="info-chip"><div class="ic-label">Area</div><div class="ic-val">${parseFloat(area).toLocaleString()} ha</div></div>
          <div class="info-chip"><div class="ic-label">Total Est. Yield</div><div class="ic-val">${(data.predicted_yield * parseFloat(area) / 1000).toFixed(1)} MT</div></div>
        </div>
      </div>`;

    document.getElementById('yield-price-section').style.display = 'block';
  } catch(e) {
    resultEl.innerHTML = `<div class="error-msg"> Failed to connect to backend. Make sure Flask server is running on port 5000.</div>`;
  }

  btn.disabled = false; btn.textContent = 'Predict Yield';
}

// ============================
// PREDICT PRICE (direct)
// ============================
async function predictPrice(fromYield) {
  let payload, crop, state, season;

  if (fromYield && yieldData) {
    crop = yieldData.crop;
    state = yieldData.state;
    season = yieldData.season;
    payload = { crop, state, season, yield_value: yieldData.yield_value };
  } else {
    crop = document.getElementById('price-crop').value;
    state = document.getElementById('price-state').value;
    season = document.getElementById('price-season').value;
    const qty = document.getElementById('price-qty').value;
    if (!crop || !state || !qty) {
      document.getElementById('price-error').innerHTML = '<div class="error-msg">⚠ Please fill all fields.</div>';
      return;
    }
    document.getElementById('price-error').innerHTML = '';
    payload = { crop, state, season, yield_value: parseFloat(qty) };
  }

  const resultId = fromYield ? 'price-from-yield-result' : 'price-result';
  const btnId = fromYield ? null : 'price-btn';

  if (btnId) { const b = document.getElementById(btnId); b.disabled=true; b.textContent='Predicting...'; }

  const resultEl = document.getElementById(resultId);
  resultEl.style.display = 'block';
  resultEl.innerHTML = `<div class="loader-wrap"><div class="spinner"></div><div class="loader-text">Running price model…</div></div>`;

  try {
    const res = await fetch(API + '/predict_price', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    // Step update for yield flow
    if (fromYield) document.getElementById('ystep3') && (document.getElementById('ystep3').className = 'step active');

    resultEl.innerHTML = `
      <div class="result-card">
        <div class="result-label">Predicted Market Price</div>
        <div class="result-value price-value">₹${data.predicted_price.toLocaleString('en-IN', {maximumFractionDigits:2})}</div>
        <div class="result-unit">per quintal (Rs./Quintal)</div>
        <div class="model-badge">✓ ${data.model_used === 'global' ? 'Global ensemble model' : data.model_used + ' model'}</div>

        <div class="result-label" style="margin-bottom:10px">Prediction Details</div>
        <div class="info-grid">
          <div class="info-chip"><div class="ic-label">Crop</div><div class="ic-val">${crop}</div></div>
          <div class="info-chip"><div class="ic-label">State</div><div class="ic-val">${state}</div></div>
          <div class="info-chip"><div class="ic-label">Season</div><div class="ic-val">${season}</div></div>
          <div class="info-chip"><div class="ic-label">Arrival Proxy</div><div class="ic-val">${data.arrival_proxy.toFixed(3)} MT</div></div>
          <div class="info-chip"><div class="ic-label">Price/MT</div><div class="ic-val">₹${(data.predicted_price * 10).toLocaleString('en-IN', {maximumFractionDigits:0})}</div></div>
        </div>
      </div>`;
  } catch(e) {
    resultEl.innerHTML = `<div class="error-msg"> Failed to connect to backend. Make sure Flask server is running on port 5000.</div>`;
  }

  if (btnId) { const b = document.getElementById(btnId); b.disabled=false; b.textContent='Predict Market Price'; }
}

// ============================
// PRICE FROM YIELD FLOW
// ============================
function loadPriceFromYield() {
  if (!yieldData) { showPage('yield'); return; }

  const summary = document.getElementById('yield-summary-card');
  summary.innerHTML = `
    <div class="form-card" style="border-color:rgba(74,222,128,0.2)">
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px">
        <div style="width:8px;height:8px;border-radius:50%;background:var(--green)"></div>
        <span style="font-size:0.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--green)">Yield carried forward</span>
      </div>
      <div class="info-grid">
        <div class="info-chip"><div class="ic-label">Crop</div><div class="ic-val">${yieldData.crop}</div></div>
        <div class="info-chip"><div class="ic-label">State</div><div class="ic-val">${yieldData.state}</div></div>
        <div class="info-chip"><div class="ic-label">Predicted Yield</div><div class="ic-val">${yieldData.yield_value.toFixed(2)} kg/ha</div></div>
      </div>
    </div>`;

  const resultEl = document.getElementById('price-from-yield-result');
  resultEl.innerHTML = `<div class="loader-wrap"><div class="spinner"></div><div class="loader-text">Running price model…</div></div>`;
  predictPrice(true);
}

// ============================
// INIT
// ============================
loadMeta();