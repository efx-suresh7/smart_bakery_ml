/* ==========================================================================
   Smart Bakery — Dashboard JS
   Handles: tab navigation, API calls, Chart.js rendering, CSV upload, date predictions,
            POS sales, customer insights, reviews & sentiment analysis, wastage
   ========================================================================== */

'use strict';

// -- Chart registry (destroy before recreate) --
const charts = {};

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// -- Shared Chart.js defaults --
Chart.defaults.color          = '#64748b';
Chart.defaults.borderColor    = '#e2e8f0';
Chart.defaults.font.family    = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size      = 12;

const PALETTE = {
  lr:      '#3b82f6',
  rf:      '#10b981',
  amber:   '#f59e0b',
  pink:    '#8b5cf6',
  items:   ['#3b82f6','#10b981','#f59e0b','#8b5cf6','#06b6d4',
             '#f97316','#84cc16','#ec4899','#14b8a6','#a855f7'],
};

function hexAlpha(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16),
        g = parseInt(hex.slice(3, 5), 16),
        b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

// -- Tab navigation --
const TAB_TITLES = {
  dashboard:   'Dashboard',
  predictions: 'Predictions',
  models:      'Model Comparison',
  products:    'Products',
  inventory:   'Inventory / Wastage',
  pos:         'POS Sales Entry',
  insights:    'Customer Insights',
  feedback:    'Feedback & Sentiment',
};

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    const tab = item.dataset.tab;
    switchTab(tab);
  });
});

function switchTab(tab) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');

  document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');

  document.getElementById('pageTitle').textContent = TAB_TITLES[tab] || tab;

  // Lazy-load tab data
  if (tab === 'models')      loadModels();
  if (tab === 'products')    loadProducts();
  if (tab === 'inventory')   { loadInventory(); loadWastageReport(); }
  if (tab === 'predictions') loadPredictions();
  if (tab === 'pos')         loadPOS();
  if (tab === 'insights')    loadInsights();
  if (tab === 'feedback')    loadFeedback();
}

// -- Fetch wrapper --
async function api(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// ==========================================================================
// LOAD ALL (on page ready + refresh button)
// ==========================================================================
async function loadAll() {
  setStatus('loading');
  try {
    await Promise.all([loadSummary(), loadMonthly(), loadWeekly(),
                       loadDaypart(), loadRecent()]);
    setStatus('online');
  } catch (err) {
    console.error(err);
    setStatus('error');
  }
}

// -- Summary / KPI cards --
async function loadSummary() {
  const d = await api('/api/summary');
  document.getElementById('pageDate').textContent = d.today || '';
  if (d.active_dataset) {
    document.getElementById('activeDataset').textContent = d.active_dataset;
  }
  animateCount('kpiTransactions', d.total_records);
  animateCount('kpiItems',        d.avg_items_per_day, 1);
  animateCount('kpiAvgDay',       d.avg_items_per_day, 1);
  animateCount('kpiPredRF',       d.predicted_today_rf, 1);
}

function animateCount(id, target, decimals = 0) {
  const el   = document.getElementById(id);
  const start = parseFloat(el.textContent) || 0;
  const diff  = target - start;
  const dur   = 800;
  const t0    = performance.now();
  const step  = (now) => {
    const p = Math.min((now - t0) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = (start + diff * eased).toFixed(decimals);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// -- Monthly trend --
async function loadMonthly() {
  const data = await api('/api/monthly');
  const labels = data.map(r => r.label);
  const values = data.map(r => r.total_items);

  destroyChart('monthly');
  const ctx = document.getElementById('chartMonthly').getContext('2d');
  charts['monthly'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Items Sold',
        data: values,
        borderColor: PALETTE.lr,
        backgroundColor: hexAlpha(PALETTE.lr, 0.08),
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 6,
        fill: true,
        tension: 0.4,
      }],
    },
    options: lineOpts('Items Sold'),
  });
}

// -- Weekly average --
async function loadWeekly() {
  const data = await api('/api/weekly');
  const labels = data.map(r => r.label);
  const values = data.map(r => +r.total_items.toFixed(1));

  destroyChart('weekly');
  const ctx = document.getElementById('chartWeekly').getContext('2d');
  charts['weekly'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Avg Items',
        data: values,
        backgroundColor: labels.map((_, i) =>
          hexAlpha(i >= 5 ? PALETTE.rf : PALETTE.lr, 0.75)),
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: barOpts('Avg Items'),
  });
}

// -- Daypart doughnut --
async function loadDaypart() {
  const data = await api('/api/daypart');
  const labels = Object.keys(data);
  const values = Object.values(data);

  destroyChart('daypart');
  const ctx = document.getElementById('chartDaypart').getContext('2d');
  charts['daypart'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: [PALETTE.lr, PALETTE.rf, PALETTE.amber, PALETTE.pink],
        borderWidth: 0,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: { position: 'bottom', labels: { color: '#475569', padding: 14, boxWidth: 12 } },
        tooltip: {
          backgroundColor: '#ffffff',
          borderColor: '#cbd5e1',
          borderWidth: 1,
          titleColor: '#1e293b',
          bodyColor: '#475569',
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed.toLocaleString()}`,
          }
        },
      },
    },
  });
}

// -- Recent 30-day line --
async function loadRecent() {
  const data = await api('/api/recent?n=30');
  const labels = data.map(r => r.date);
  const values = data.map(r => r.total_items);

  destroyChart('recent');
  const ctx = document.getElementById('chartRecent').getContext('2d');
  charts['recent'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Items Sold',
        data: values,
        borderColor: PALETTE.rf,
        backgroundColor: hexAlpha(PALETTE.rf, 0.05),
        borderWidth: 2,
        pointRadius: 2,
        fill: true,
        tension: 0.35,
      }],
    },
    options: lineOpts('Items'),
  });
}

// ==========================================================================
// POS SALES ENTRY SYSTEM
// ==========================================================================
let currentSale = {};

function loadPOS() {
  currentSale = { Coffee: 0, Bread: 0, Tea: 0, Cake: 0, Pastry: 0 };
  updatePOSUI();
}

function changePOSQty(item, amount) {
  if (currentSale[item] === undefined) currentSale[item] = 0;
  currentSale[item] = Math.max(0, currentSale[item] + amount);
  document.getElementById(`pos_${item}`).value = currentSale[item];
  updatePOSUI();
}

function clearPOS() {
  loadPOS();
  document.querySelectorAll('#tab-pos input[type="number"]').forEach(input => input.value = 0);
}

function updatePOSUI() {
  const list = document.getElementById('posSummaryList');
  list.innerHTML = '';
  
  let itemsList = [];
  let totalItems = 0;
  
  Object.entries(currentSale).forEach(([item, qty]) => {
    if (qty > 0) {
      itemsList.push({ item, qty });
      totalItems += qty;
    }
  });
  
  if (itemsList.length === 0) {
    list.innerHTML = `<p style="color: var(--text-muted); font-size: 0.88rem;">No items added to current sale.</p>`;
    return;
  }
  
  itemsList.forEach(({ item, qty }) => {
    list.innerHTML += `
      <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--card-border); padding-bottom:6px;">
        <span style="font-weight:500;">${item}</span>
        <span style="color:var(--text-dim);">Qty: <strong>${qty}</strong></span>
      </div>
    `;
  });
  
  list.innerHTML += `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px; font-weight:600; font-size:1.05rem;">
      <span>Total Items:</span>
      <span>${totalItems}</span>
    </div>
  `;
}

async function submitPOSSale() {
  let itemsList = [];
  Object.entries(currentSale).forEach(([item, qty]) => {
    for (let i = 0; i < qty; i++) {
      itemsList.push(item);
    }
  });
  
  if (itemsList.length === 0) {
    alert('Please add items to record first.');
    return;
  }
  
  setStatus('loading');
  try {
    const res = await fetch('/api/sales', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items: itemsList })
    });
    
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to record transaction');
    
    alert(data.message);
    clearPOS();
    await loadAll();
  } catch (err) {
    console.error(err);
    alert(err.message);
    setStatus('error');
  }
}

// ==========================================================================
// PREDICTIONS
// ==========================================================================
async function loadPredictions() {
  const days   = document.getElementById('predDays').value;
  const target = document.getElementById('predTarget').value;

  const data = await api(`/api/predict?days=${days}&target=${target}`);

  destroyChart('pred');
  const ctx = document.getElementById('chartPred').getContext('2d');
  charts['pred'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [
        {
          label: 'Linear Regression',
          data: data.linear_regression,
          borderColor: PALETTE.lr,
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 4,
          fill: false,
          tension: 0.3,
        },
        {
          label: 'Random Forest',
          data: data.random_forest,
          borderColor: PALETTE.rf,
          backgroundColor: hexAlpha(PALETTE.rf, 0.05),
          borderWidth: 2,
          pointRadius: 4,
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: lineOpts(target === 'total_items' ? 'Items' : 'Transactions', true),
  });

  // Summary cards below chart
  const lrTotal = data.linear_regression.reduce((a, b) => a + b, 0);
  const rfTotal = data.random_forest.reduce((a, b) => a + b, 0);
  const lrAvg   = lrTotal / data.linear_regression.length;
  const rfAvg   = rfTotal / data.random_forest.length;

  document.getElementById('predSummaryCards').innerHTML = `
    <div class="pred-summary-card">
      <div class="ps-label">LR Total (${days} days)</div>
      <div class="ps-val" style="color:${PALETTE.lr}">${Math.round(lrTotal).toLocaleString()}</div>
      <div class="ps-model">Linear Regression</div>
    </div>
    <div class="pred-summary-card">
      <div class="ps-label">RF Total (${days} days)</div>
      <div class="ps-val" style="color:${PALETTE.rf}">${Math.round(rfTotal).toLocaleString()}</div>
      <div class="ps-model">Random Forest</div>
    </div>
    <div class="pred-summary-card">
      <div class="ps-label">LR Daily Avg</div>
      <div class="ps-val" style="color:${PALETTE.lr}">${lrAvg.toFixed(1)}</div>
      <div class="ps-model">Linear Regression</div>
    </div>
    <div class="pred-summary-card">
      <div class="ps-label">RF Daily Avg</div>
      <div class="ps-val" style="color:${PALETTE.rf}">${rfAvg.toFixed(1)}</div>
      <div class="ps-model">Random Forest</div>
    </div>
  `;
}

// Predict specific date
async function predictSingleDate() {
  const dateInput = document.getElementById('singlePredDate').value;
  const target = document.getElementById('singlePredTarget').value;
  const resultsDiv = document.getElementById('singlePredResults');
  
  if (!dateInput) {
    alert('Please select a date first.');
    return;
  }
  
  try {
    setStatus('loading');
    const data = await api(`/api/predict-date?date=${dateInput}&target=${target}`);
    setStatus('online');
    
    const lrVal = data.linear_regression;
    const rfVal = data.random_forest;
    
    document.getElementById('singleResultLR').textContent = Math.round(lrVal).toLocaleString();
    document.getElementById('singleResultRF').textContent = Math.round(rfVal).toLocaleString();
    
    const targetLabel = target === 'total_items' ? 'items' : 'transactions';
    const dayType = data.is_weekend ? 'Weekend' : 'Weekday';
    const holType = data.is_holiday ? ', Holiday' : '';
    
    document.getElementById('singleMetaLR').textContent = `Linear prediction (${targetLabel}) for typical ${dayType}`;
    document.getElementById('singleMetaRF').textContent = `Forest prediction (${targetLabel}) for ${dayType}${holType}`;
    
    resultsDiv.style.display = 'grid';
  } catch (err) {
    console.error(err);
    alert(err.message);
    setStatus('error');
  }
}

// ==========================================================================
// MODEL COMPARISON
// ==========================================================================
async function loadModels() {
  const [metrics, fi] = await Promise.all([
    api('/api/metrics'),
    api('/api/feature-importance'),
  ]);

  // -- R² and MAE bar charts --
  const targets     = Object.keys(metrics);
  const r2_lr       = targets.map(t => metrics[t].linear_regression.r2);
  const r2_rf       = targets.map(t => metrics[t].random_forest.r2);
  const mae_lr      = targets.map(t => metrics[t].linear_regression.mae);
  const mae_rf      = targets.map(t => metrics[t].random_forest.mae);
  const tLabels     = targets.map(t => t.replace('_', ' '));

  destroyChart('r2');
  const ctxR2 = document.getElementById('chartR2').getContext('2d');
  charts['r2'] = new Chart(ctxR2, {
    type: 'bar',
    data: {
      labels: tLabels,
      datasets: [
        { label: 'Linear Regression', data: r2_lr, backgroundColor: hexAlpha(PALETTE.lr, 0.75), borderRadius: 4, borderSkipped: false },
        { label: 'Random Forest',     data: r2_rf, backgroundColor: hexAlpha(PALETTE.rf, 0.75), borderRadius: 4, borderSkipped: false },
      ],
    },
    options: { ...barOpts('R2 Score'), scales: { y: { min: 0, max: 1, ...gridStyle() } } },
  });

  destroyChart('mae');
  const ctxMAE = document.getElementById('chartMAE').getContext('2d');
  charts['mae'] = new Chart(ctxMAE, {
    type: 'bar',
    data: {
      labels: tLabels,
      datasets: [
        { label: 'Linear Regression', data: mae_lr, backgroundColor: hexAlpha(PALETTE.lr, 0.75), borderRadius: 4, borderSkipped: false },
        { label: 'Random Forest',     data: mae_rf, backgroundColor: hexAlpha(PALETTE.rf, 0.75), borderRadius: 4, borderSkipped: false },
      ],
    },
    options: barOpts('MAE'),
  });

  // -- Feature importance bar --
  destroyChart('feat');
  const fiLabels = Object.keys(fi).map(k => k.replace('_', ' '));
  const fiValues = Object.values(fi);
  const ctxFeat  = document.getElementById('chartFeat').getContext('2d');
  charts['feat'] = new Chart(ctxFeat, {
    type: 'bar',
    data: {
      labels: fiLabels,
      datasets: [{
        label: 'Importance',
        data: fiValues,
        backgroundColor: fiValues.map((_, i) =>
          hexAlpha(PALETTE.items[i % PALETTE.items.length], 0.8)),
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: { ...barOpts('Feature Importance'), indexAxis: 'y' },
  });

  // -- Metrics table --
  const tbody = document.getElementById('metricsBody');
  tbody.innerHTML = '';
  for (const [target, models] of Object.entries(metrics)) {
    for (const [mkey, m] of Object.entries(models)) {
      const cls   = mkey === 'linear_regression' ? 'model-lr' : 'model-rf';
      const label = mkey === 'linear_regression' ? 'Linear Regression' : 'Random Forest';
      const acc   = Math.min(m.accuracy, 100);
      tbody.innerHTML += `
        <tr>
          <td>${target.replace('_', ' ')}</td>
          <td><span class="model-badge ${cls}">${label}</span></td>
          <td>${m.mae}</td>
          <td>${m.rmse}</td>
          <td>${m.r2}</td>
          <td>
            <div class="acc-bar">
              <div class="acc-track">
                <div class="acc-fill" style="width:${acc}%"></div>
              </div>
              ${acc.toFixed(1)}%
            </div>
          </td>
        </tr>`;
    }
  }
}

// ==========================================================================
// CUSTOMER INSIGHTS (SEGMENTS + COMBOS)
// ==========================================================================
async function loadInsights() {
  setStatus('loading');
  try {
    const data = await api('/api/customer-insights');
    
    // 1. Render K-Means segments Chart
    destroyChart('segments');
    const ctx = document.getElementById('chartSegments').getContext('2d');
    
    const labels = data.segments.map(s => s.label);
    const counts = data.segments.map(s => s.count);
    
    charts['segments'] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: counts,
          backgroundColor: [PALETTE.lr, PALETTE.rf, PALETTE.amber],
          borderWidth: 0,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '60%',
        plugins: {
          legend: { position: 'right', labels: { color: '#475569', boxWidth: 12 } },
          tooltip: {
            backgroundColor: '#ffffff',
            borderColor: '#cbd5e1',
            borderWidth: 1,
            titleColor: '#1e293b',
            bodyColor: '#475569'
          }
        }
      }
    });
    
    // Render segments details list
    const details = document.getElementById('segmentDetailsList');
    details.innerHTML = data.segments.map(s => `
      <div style="display:flex; flex-direction:column; gap:4px; padding:10px 14px; background:var(--bg2); border-radius:var(--radius-sm); margin-bottom:8px; border:1px solid var(--card-border);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:600; color:var(--text);">${s.label}</span>
          <span class="badge badge-blue">${s.count} transactions</span>
        </div>
        <div style="font-size:0.78rem; color:var(--text-muted); display:flex; gap:12px; margin-top:4px;">
          <span>Avg items: <strong>${s.avg_items}</strong></span>
          <span>Typical time: <strong>${s.avg_hour}</strong></span>
          <span>Coffee ratio: <strong>${s.coffee_pct}%</strong></span>
        </div>
      </div>
    `).join('');
    
    // 2. Render Apriori Combos table
    const combosBody = document.getElementById('combosTableBody');
    if (data.combos.length === 0) {
      combosBody.innerHTML = `<tr><td colspan="4" style="text-align:center;">No combos mined.</td></tr>`;
    } else {
      combosBody.innerHTML = data.combos.map(c => `
        <tr>
          <td style="font-weight:600; color:var(--text);">${c.item1} + ${c.item2}</td>
          <td>${c.support}%</td>
          <td>
            <div style="display:flex; flex-direction:column; gap:2px; font-size:0.75rem;">
              <span>${c.item1} &rarr; ${c.item2}: <strong>${c.confidence_1_to_2}%</strong></span>
              <span>${c.item2} &rarr; ${c.item1}: <strong>${c.confidence_2_to_1}%</strong></span>
            </div>
          </td>
          <td>${c.count} times</td>
        </tr>
      `).join('');
    }
    
    setStatus('online');
  } catch (err) {
    console.error(err);
    setStatus('error');
  }
}

// ==========================================================================
// PRODUCTS
// ==========================================================================
async function loadProducts() {
  const data = await api('/api/top-items?n=10');
  const labels = Object.keys(data);
  const values = Object.values(data);
  const maxVal = Math.max(...values);

  // Bar chart
  destroyChart('topItems');
  const ctx1 = document.getElementById('chartTopItems').getContext('2d');
  charts['topItems'] = new Chart(ctx1, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Total Sold',
        data: values,
        backgroundColor: labels.map((_, i) => hexAlpha(PALETTE.items[i % PALETTE.items.length], 0.8)),
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: { ...barOpts('Items Sold'), indexAxis: 'y' },
  });

  // Pie / doughnut
  destroyChart('topPie');
  const ctx2 = document.getElementById('chartTopPie').getContext('2d');
  charts['topPie'] = new Chart(ctx2, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: PALETTE.items.map(c => hexAlpha(c, 0.85)),
        borderWidth: 0,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: { position: 'bottom', labels: { color: '#475569', padding: 10, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          backgroundColor: '#ffffff',
          borderColor: '#cbd5e1',
          borderWidth: 1,
          titleColor: '#1e293b',
          bodyColor: '#475569',
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed.toLocaleString()}`
          }
        },
      },
    },
  });

  // Rankings list
  document.getElementById('productRankings').innerHTML = labels.map((item, i) => `
    <div class="product-row">
      <div class="product-rank">#${i + 1}</div>
      <div class="product-name">${item}</div>
      <div class="product-bar-wrap">
        <div class="product-bar" style="width:${(values[i]/maxVal*100).toFixed(1)}%;
          background: ${PALETTE.items[i % PALETTE.items.length]}">
        </div>
      </div>
      <div class="product-count">${values[i].toLocaleString()}</div>
    </div>
  `).join('');
}

// ==========================================================================
// INVENTORY & WASTAGE
// ==========================================================================
async function loadInventory() {
  const data = await api('/api/inventory');
  const stock   = data.stock;
  const alerts  = data.alerts;

  // Stock inputs
  document.getElementById('inventoryInputs').innerHTML =
    Object.entries(stock).map(([item, qty]) => `
      <div class="inventory-item">
        <label for="stock_${item}">${item}</label>
        <input type="number" id="stock_${item}" value="${qty}" min="0" />
      </div>
    `).join('');

  renderAlerts(alerts);
}

async function saveInventory() {
  const inputs = document.querySelectorAll('#inventoryInputs input');
  const stock  = {};
  inputs.forEach(inp => {
    const key = inp.id.replace('stock_', '');
    stock[key] = parseInt(inp.value) || 0;
  });

  const res  = await fetch('/api/inventory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stock }),
  });
  const data = await res.json();
  renderAlerts(data.alerts);
  
  // Reload wastage report if stock numbers change
  await loadWastageReport();
}

function renderAlerts(alerts) {
  document.getElementById('inventoryAlerts').innerHTML = alerts.map(a => `
    <div class="alert-row alert-${a.status}">
      <div class="alert-name">${a.product}</div>
      <div class="alert-detail">
        Stock: <strong>${a.current_stock}</strong> &nbsp;|&nbsp;
        7-day demand: <strong>${a.predicted_demand}</strong> &nbsp;|&nbsp;
        Coverage: <strong>${a.days_coverage} days</strong>
      </div>
      <div class="alert-status status-${a.status}">${a.status}</div>
    </div>
  `).join('');
}

async function loadWastageReport() {
  const hour = document.getElementById('wastageHour').value;
  try {
    const data = await api(`/api/wastage?hour=${hour}`);
    const report = data.report;
    
    const body = document.getElementById('wastageTableBody');
    body.innerHTML = report.map(r => {
      const badgeClass = r.predicted_waste > 5 ? 'badge-purple' : r.predicted_waste > 0 ? 'badge-amber' : 'badge-green';
      
      return `
        <tr>
          <td style="font-weight:600; color:var(--text);">${r.product}</td>
          <td>${r.current_stock}</td>
          <td>${r.expected_demand_today}</td>
          <td>${r.expected_sold_so_far}</td>
          <td><span class="badge ${badgeClass}">${r.predicted_waste} units</span></td>
          <td>${r.waste_percentage}%</td>
          <td><span class="badge ${badgeClass}" style="text-transform:none;">${r.discount_action}</span></td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error(err);
  }
}

// ==========================================================================
// FEEDBACK & SENTIMENT ANALYSIS
// ==========================================================================
async function loadFeedback() {
  setStatus('loading');
  try {
    const data = await api('/api/reviews');
    
    // Render reviews list
    const list = document.getElementById('reviewsList');
    list.innerHTML = data.reviews.map(r => {
      const sentClass = r.sentiment === 'positive' ? 'status-ok' : r.sentiment === 'negative' ? 'status-critical' : 'status-low';
      const stars = '★'.repeat(r.rating) + '☆'.repeat(5 - r.rating);
      const flagBadges = r.flags.map(f => `<span class="badge badge-purple" style="font-size:0.65rem; padding:1px 4px; margin-left:4px; text-transform:none;">${f}</span>`).join('');
      
      return `
        <div style="padding:12px 14px; background:var(--bg); border:1px solid var(--card-border); border-radius:var(--radius-sm); position:relative; margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <strong style="color:var(--text); font-size:0.9rem;">${r.product}</strong>
            <span class="alert-status ${sentClass}" style="font-size:0.68rem; padding:2px 6px;">${r.sentiment}</span>
          </div>
          <div style="color:var(--accent3); font-size:0.8rem; margin-bottom:4px;">${stars}</div>
          <p style="font-size:0.85rem; color:var(--text-dim); line-height:1.4;">${r.comment}</p>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.75rem; color:var(--text-muted);">
            <span>${r.date}</span>
            <div>${flagBadges}</div>
          </div>
        </div>
      `;
    }).join('');
    
    // Render alerts list
    const alertsList = document.getElementById('sentimentAlertsList');
    if (data.alerts.length === 0) {
      alertsList.innerHTML = `<p style="color:var(--text-muted); font-size:0.82rem; text-align:center;">No quality issues flagged. Sentiment is positive!</p>`;
    } else {
      alertsList.innerHTML = data.alerts.map(a => `
        <div class="alert-row alert-critical" style="padding:10px 12px; border-left-width:4px;">
          <div style="display:flex; flex-direction:column; gap:4px; width: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
              <strong style="font-size:0.88rem; color:var(--text);">${a.product}</strong>
              <span class="badge badge-purple" style="font-size:0.68rem;">${a.negative_percentage}% negative</span>
            </div>
            <span style="font-size:0.76rem; color:var(--text-dim);">From ${a.total_reviews} reviews</span>
            <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:4px;">
              ${a.key_issues.map(issue => `<span class="badge badge-amber" style="font-size:0.62rem; padding:1px 4px; text-transform:none;">${issue}</span>`).join('')}
            </div>
          </div>
        </div>
      `).join('');
    }
    
    setStatus('online');
  } catch (err) {
    console.error(err);
    setStatus('error');
  }
}

async function submitReview() {
  const product = document.getElementById('reviewProduct').value;
  const rating = document.getElementById('reviewRating').value;
  const comment = document.getElementById('reviewComment').value.trim();
  
  if (!comment) {
    alert('Please enter review comment text first.');
    return;
  }
  
  setStatus('loading');
  try {
    const res = await fetch('/api/reviews', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product, rating, comment })
    });
    
    if (!res.ok) throw new Error('Failed to submit review');
    
    document.getElementById('reviewComment').value = '';
    alert('Review submitted successfully.');
    await loadFeedback();
  } catch (err) {
    console.error(err);
    alert(err.message);
    setStatus('error');
  }
}

// ==========================================================================
// CSV UPLOAD
// ==========================================================================
async function uploadCSV(input) {
  if (!input.files || input.files.length === 0) return;
  const file = input.files[0];
  
  const formData = new FormData();
  formData.append('file', file);
  
  setStatus('loading');
  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'Failed to upload CSV');
    }
    
    document.getElementById('activeDataset').textContent = data.active_dataset;
    input.value = '';
    
    // Reload all graphs and data
    await loadAll();
    alert(data.message);
  } catch (err) {
    console.error(err);
    alert(err.message);
    setStatus('error');
  }
}

// ==========================================================================
// CHART OPTION PRESETS
// ==========================================================================
function gridStyle() {
  return { grid: { color: '#e2e8f0' }, ticks: { color: '#64748b' } };
}

function lineOpts(yLabel = '', legend = false) {
  return {
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: legend, labels: { color: '#475569', padding: 14, boxWidth: 14 } },
      tooltip: {
        backgroundColor: '#ffffff',
        borderColor: '#cbd5e1',
        borderWidth: 1,
        padding: 12,
        titleColor: '#1e293b',
        bodyColor: '#475569',
      },
    },
    scales: {
      x: { ...gridStyle(), ticks: { color: '#64748b', maxRotation: 45 } },
      y: { ...gridStyle(), title: { display: !!yLabel, text: yLabel, color: '#64748b' } },
    },
  };
}

function barOpts(yLabel = '') {
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: true, labels: { color: '#475569', padding: 14, boxWidth: 14 } },
      tooltip: {
        backgroundColor: '#ffffff',
        borderColor: '#cbd5e1',
        borderWidth: 1,
        padding: 12,
        titleColor: '#1e293b',
        bodyColor: '#475569',
      },
    },
    scales: {
      x: { ...gridStyle() },
      y: { ...gridStyle(), title: { display: !!yLabel, text: yLabel, color: '#64748b' } },
    },
  };
}

// ==========================================================================
// STATUS INDICATOR
// ==========================================================================
function setStatus(state) {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  if (state === 'online') {
    dot.className = 'status-dot online';
    text.textContent = 'Models ready';
  } else if (state === 'loading') {
    dot.className = 'status-dot';
    text.innerHTML = '<span class="loader"></span> Training...';
  } else {
    dot.className = 'status-dot';
    text.textContent = 'Error - check logs';
  }
}

// ==========================================================================
// INIT
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  setStatus('loading');
  loadAll();
});
