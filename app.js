/* ═══════════════════════════════════════════════════════════════════════════
   Resume Analysis Dashboard — App Logic
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  /* ── Color Palettes ──────────────────────────────────────────────────── */
  const COLORS = {
    blue:    '#3b82f6',
    purple:  '#8b5cf6',
    pink:    '#ec4899',
    green:   '#10b981',
    orange:  '#f59e0b',
    red:     '#ef4444',
    cyan:    '#06b6d4',
    indigo:  '#6366f1',
    teal:    '#14b8a6',
  };

  const MODEL_COLORS = [
    '#3b82f6', '#8b5cf6', '#ec4899', '#10b981',
    '#f59e0b', '#ef4444', '#06b6d4', '#6366f1', '#14b8a6',
  ];

  const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
      }
    },
    scales: {
      x: {
        ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } },
        grid:  { color: 'rgba(255,255,255,0.04)' }
      },
      y: {
        ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } },
        grid:  { color: 'rgba(255,255,255,0.04)' }
      }
    }
  };

  /* ── Utilities ───────────────────────────────────────────────────────── */
  function heatColor(val, min, max) {
    const ratio = (val - min) / (max - min || 1);
    if (ratio >= 0.85) return 'rgba(16, 185, 129, 0.55)';
    if (ratio >= 0.7)  return 'rgba(16, 185, 129, 0.3)';
    if (ratio >= 0.5)  return 'rgba(245, 158, 11, 0.3)';
    if (ratio >= 0.3)  return 'rgba(245, 158, 11, 0.2)';
    return 'rgba(239, 68, 68, 0.25)';
  }

  function errorColor(val) {
    if (val === 0) return COLORS.green;
    if (val < 5)   return COLORS.teal;
    if (val < 10)  return COLORS.orange;
    if (val < 15)  return COLORS.pink;
    return COLORS.red;
  }

  function metricClass(val) {
    if (val >= 0.95) return 'high';
    if (val >= 0.90) return 'medium';
    return 'low';
  }

  function animateCounter(el, target, decimals, duration) {
    const start = 0;
    const startTime = performance.now();
    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = start + (target - start) * eased;
      el.textContent = decimals ? current.toFixed(decimals) : Math.round(current).toLocaleString();
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* ── Load Data & Render ──────────────────────────────────────────────── */
  fetch('analysis_results.json')
    .then(r => r.json())
    .then(data => renderDashboard(data))
    .catch(err => {
      console.error('Failed to load data:', err);
      document.querySelector('main').innerHTML =
        '<div class="card" style="text-align:center;padding:4rem"><h3>⚠️ Could not load analysis_results.json</h3><p style="color:var(--text-muted);margin-top:1rem">Make sure the file exists in the same directory as index.html</p></div>';
    });

  function renderDashboard(data) {
    renderHeroStats(data);
    renderModelTable(data);
    renderModelBarChart(data);
    renderRadarChart(data);
    renderHeatmap(data);
    renderConfusion(data);
    renderDistribution(data);
    renderErrors(data);
    renderSkillMatrix(data);
    renderVocabulary(data);
    renderAblation(data);
    setupScrollTop();
    setupNavHighlight();
  }

  /* ── 1. Hero Stats ───────────────────────────────────────────────────── */
  function renderHeroStats(data) {
    const container = document.getElementById('hero-stats');
    const bestModel = data.best_model;
    const bestF1 = data.model_comparison[bestModel]?.F1 || 0;
    const totalModels = Object.keys(data.model_comparison).length;

    const stats = [
      { icon: '📄', value: data.dataset_info.total_samples, label: 'Total Resumes', decimals: 0 },
      { icon: '🏷️', value: data.dataset_info.num_categories, label: 'Job Categories', decimals: 0 },
      { icon: '🤖', value: totalModels, label: 'Models Trained', decimals: 0 },
      { icon: '🏆', value: bestF1 * 100, label: 'Best F1 (%)', decimals: 2 },
    ];

    stats.forEach((s, i) => {
      const card = document.createElement('div');
      card.className = 'stat-card animate-in';
      card.style.animationDelay = `${i * 0.1}s`;
      card.innerHTML = `
        <div class="stat-icon">${s.icon}</div>
        <div class="stat-value" id="stat-val-${i}">0</div>
        <div class="stat-label">${s.label}</div>
      `;
      container.appendChild(card);
      setTimeout(() => {
        animateCounter(card.querySelector('.stat-value'), s.value, s.decimals, 1200);
      }, 300 + i * 150);
    });
  }

  /* ── 2. Model Comparison Table ───────────────────────────────────────── */
  function renderModelTable(data) {
    const tbody = document.getElementById('model-table-body');
    const models = data.model_comparison;
    const bestModel = data.best_model;

    const sorted = Object.entries(models).sort((a, b) => b[1].F1 - a[1].F1);

    sorted.forEach(([name, m]) => {
      const tr = document.createElement('tr');
      const isBest = name === bestModel;
      tr.innerHTML = `
        <td class="model-name">
          ${name}
          ${isBest ? '<span class="best-badge">🏆 BEST</span>' : ''}
        </td>
        <td class="metric-cell ${metricClass(m.Accuracy)}">${(m.Accuracy * 100).toFixed(2)}%</td>
        <td class="metric-cell ${metricClass(m.Precision)}">${(m.Precision * 100).toFixed(2)}%</td>
        <td class="metric-cell ${metricClass(m.Recall)}">${(m.Recall * 100).toFixed(2)}%</td>
        <td class="metric-cell ${metricClass(m.F1)}">${(m.F1 * 100).toFixed(2)}%</td>
        <td class="metric-cell">${m.Time_s > 0 ? m.Time_s.toFixed(2) : '—'}</td>
      `;
      tbody.appendChild(tr);
    });

    // Sortable headers
    const table = document.getElementById('model-table');
    let sortDir = {};
    table.querySelectorAll('thead th').forEach(th => {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        sortDir[key] = !sortDir[key];
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {
          let valA, valB;
          if (key === 'name') {
            valA = a.cells[0].textContent.trim();
            valB = b.cells[0].textContent.trim();
            return sortDir[key] ? valA.localeCompare(valB) : valB.localeCompare(valA);
          }
          const colIdx = { Accuracy: 1, Precision: 2, Recall: 3, F1: 4, Time_s: 5 }[key];
          valA = parseFloat(a.cells[colIdx].textContent) || 0;
          valB = parseFloat(b.cells[colIdx].textContent) || 0;
          return sortDir[key] ? valA - valB : valB - valA;
        });
        rows.forEach(r => tbody.appendChild(r));
      });
    });
  }

  /* ── 3. Model Bar Chart ──────────────────────────────────────────────── */
  function renderModelBarChart(data) {
    const models = data.model_comparison;
    const sorted = Object.entries(models).sort((a, b) => b[1].F1 - a[1].F1);
    const labels = sorted.map(([n]) => n);
    const metrics = ['Accuracy', 'Precision', 'Recall', 'F1'];
    const colors = [COLORS.blue, COLORS.green, COLORS.orange, COLORS.pink];

    new Chart(document.getElementById('model-bar-chart'), {
      type: 'bar',
      data: {
        labels,
        datasets: metrics.map((m, i) => ({
          label: m,
          data: sorted.map(([, v]) => (v[m] * 100).toFixed(2)),
          backgroundColor: colors[i] + '99',
          borderColor: colors[i],
          borderWidth: 1,
          borderRadius: 4,
        }))
      },
      options: {
        ...CHART_DEFAULTS,
        indexAxis: 'y',
        plugins: {
          ...CHART_DEFAULTS.plugins,
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${ctx.raw}%`
            }
          }
        },
        scales: {
          x: {
            ...CHART_DEFAULTS.scales.x,
            min: Math.max(0, Math.floor(Math.min(...sorted.map(([,v]) => Math.min(v.Accuracy,v.Precision,v.Recall,v.F1)*100)) / 5) * 5 - 5),
            max: Math.min(100, Math.ceil(Math.max(...sorted.map(([,v]) => Math.max(v.Accuracy,v.Precision,v.Recall,v.F1)*100)) / 5) * 5 + 5),
            title: { display: true, text: 'Score (%)', color: '#64748b' }
          },
          y: { ...CHART_DEFAULTS.scales.y }
        }
      }
    });
  }

  /* ── 4. Radar Chart ──────────────────────────────────────────────────── */
  function renderRadarChart(data) {
    const models = data.model_comparison;
    const sorted = Object.entries(models).sort((a, b) => b[1].F1 - a[1].F1);
    const top4 = sorted.slice(0, 4);
    const metrics = ['Accuracy', 'Precision', 'Recall', 'F1'];
    const colors = [COLORS.pink, COLORS.blue, COLORS.green, COLORS.orange];

    new Chart(document.getElementById('radar-chart'), {
      type: 'radar',
      data: {
        labels: metrics,
        datasets: top4.map(([name, vals], i) => ({
          label: name,
          data: metrics.map(m => (vals[m] * 100).toFixed(2)),
          borderColor: colors[i],
          backgroundColor: colors[i] + '18',
          borderWidth: 2,
          pointBackgroundColor: colors[i],
          pointRadius: 4,
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, padding: 16 }
          }
        },
        scales: {
          r: {
            min: Math.max(0, Math.floor(Math.min(...top4.flatMap(([,v]) => metrics.map(m => v[m]*100))) / 5) * 5 - 5),
            max: Math.min(100, Math.ceil(Math.max(...top4.flatMap(([,v]) => metrics.map(m => v[m]*100))) / 5) * 5 + 5),
            ticks: { display: false },
            grid: { color: 'rgba(255,255,255,0.06)' },
            angleLines: { color: 'rgba(255,255,255,0.06)' },
            pointLabels: { color: '#94a3b8', font: { size: 12, family: 'Inter' } }
          }
        }
      }
    });
  }

  /* ── 5. Per-Class Heatmap ────────────────────────────────────────────── */
  function renderHeatmap(data) {
    const pcf1 = data.per_class_f1;
    const modelNames = Object.keys(pcf1);
    const categories = Object.keys(pcf1[modelNames[0]]);

    // Collect all values for color scale
    let allVals = [];
    modelNames.forEach(m => categories.forEach(c => allVals.push(pcf1[m][c])));
    const minVal = Math.min(...allVals);
    const maxVal = Math.max(...allVals);

    // Head
    const thead = document.getElementById('heatmap-head');
    let headRow = '<tr><th class="row-header">Category</th>';
    modelNames.forEach(m => {
      const short = m.replace('Logistic Regression', 'LogReg')
                     .replace('Multinomial NB', 'MNB')
                     .replace('Gradient Boosting', 'GBM')
                     .replace('Random Forest', 'RF');
      headRow += `<th>${short}</th>`;
    });
    headRow += '<th>Best</th></tr>';
    thead.innerHTML = headRow;

    // Body
    const tbody = document.getElementById('heatmap-body');
    categories.sort().forEach(cat => {
      let row = `<tr><td class="category-label">${cat}</td>`;
      let best = 0;
      let bestModel = '';
      modelNames.forEach(m => {
        const val = pcf1[m][cat];
        if (val > best) { best = val; bestModel = m; }
        row += `<td style="background:${heatColor(val, minVal, maxVal)}" title="${m}: ${val}">${val.toFixed(3)}</td>`;
      });
      row += `<td style="font-weight:600;color:${COLORS.green};font-size:0.7rem" title="${bestModel}">${best.toFixed(3)}</td>`;
      row += '</tr>';
      tbody.innerHTML += row;
    });
  }

  /* ── 6. Confusion Analysis ───────────────────────────────────────────── */
  function renderConfusion(data) {
    const pairs = data.confusion_data.top_confused_pairs;
    const container = document.getElementById('confused-pairs-list');
    const countBadge = document.getElementById('confused-count');

    countBadge.textContent = `${pairs.length} pairs`;

    pairs.forEach((p, i) => {
      const el = document.createElement('div');
      el.className = 'confused-pair';
      el.innerHTML = `
        <span class="pair-rank">#${i + 1}</span>
        <div class="pair-labels">
          <span class="pair-actual">${p.Actual}</span>
          <span class="pair-arrow">→</span>
          <span class="pair-predicted">${p.Predicted}</span>
        </div>
        <span class="pair-count">${p.Count}</span>
        <span class="pair-pct">${p.Pct_of_actual}%</span>
      `;
      container.appendChild(el);
    });

    // Chart
    const topN = pairs.slice(0, 10);
    new Chart(document.getElementById('confusion-chart'), {
      type: 'bar',
      data: {
        labels: topN.map(p => `${p.Actual} → ${p.Predicted}`),
        datasets: [{
          label: 'Misclassifications',
          data: topN.map(p => p.Count),
          backgroundColor: topN.map((_, i) =>
            `rgba(239, 68, 68, ${0.4 + (1 - i / topN.length) * 0.5})`
          ),
          borderColor: COLORS.red,
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        ...CHART_DEFAULTS,
        indexAxis: 'y',
        plugins: {
          ...CHART_DEFAULTS.plugins,
          legend: { display: false }
        },
        scales: {
          x: {
            ...CHART_DEFAULTS.scales.x,
            title: { display: true, text: 'Count', color: '#64748b' }
          },
          y: {
            ...CHART_DEFAULTS.scales.y,
            ticks: { ...CHART_DEFAULTS.scales.y.ticks, font: { size: 9 } }
          }
        }
      }
    });
  }

  /* ── 7. Category Distribution ────────────────────────────────────────── */
  function renderDistribution(data) {
    const counts = data.dataset_info.category_counts;
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(([c]) => c);
    const values = sorted.map(([, v]) => v);

    // Generate gradient colors
    const barColors = values.map((_, i) => {
      const ratio = i / values.length;
      const r = Math.round(59 + (236 - 59) * ratio);
      const g = Math.round(130 + (72 - 130) * ratio);
      const b = Math.round(246 + (153 - 246) * ratio);
      return `rgba(${r}, ${g}, ${b}, 0.75)`;
    });

    new Chart(document.getElementById('distribution-chart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Resume Count',
          data: values,
          backgroundColor: barColors,
          borderColor: barColors.map(c => c.replace('0.75', '1')),
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        ...CHART_DEFAULTS,
        indexAxis: 'y',
        plugins: {
          ...CHART_DEFAULTS.plugins,
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.raw} resumes (${(ctx.raw / data.dataset_info.total_samples * 100).toFixed(1)}%)`
            }
          }
        },
        scales: {
          x: {
            ...CHART_DEFAULTS.scales.x,
            title: { display: true, text: 'Count', color: '#64748b' }
          },
          y: { ...CHART_DEFAULTS.scales.y }
        }
      }
    });

    // Donut
    const top8 = sorted.slice(0, 8);
    const othersCount = sorted.slice(8).reduce((s, [, v]) => s + v, 0);
    const donutLabels = [...top8.map(([c]) => c), 'Others'];
    const donutValues = [...top8.map(([, v]) => v), othersCount];
    const donutColors = [
      '#3b82f6', '#8b5cf6', '#ec4899', '#10b981',
      '#f59e0b', '#06b6d4', '#6366f1', '#14b8a6', '#64748b'
    ];

    new Chart(document.getElementById('donut-chart'), {
      type: 'doughnut',
      data: {
        labels: donutLabels,
        datasets: [{
          data: donutValues,
          backgroundColor: donutColors.map(c => c + 'cc'),
          borderColor: donutColors,
          borderWidth: 2,
          hoverOffset: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '55%',
        plugins: {
          legend: {
            position: 'right',
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, padding: 10, usePointStyle: true }
          },
          tooltip: {
            callbacks: {
              label: ctx => {
                const pct = (ctx.raw / data.dataset_info.total_samples * 100).toFixed(1);
                return `${ctx.label}: ${ctx.raw} (${pct}%)`;
              }
            }
          }
        }
      }
    });
  }

  /* ── 8. Error Analysis ───────────────────────────────────────────────── */
  function renderErrors(data) {
    const container = document.getElementById('error-bars');
    const badge = document.getElementById('best-model-badge');
    badge.textContent = data.best_model;

    const errors = data.error_rates;
    const maxErr = Math.max(...errors.map(e => e.Error_Rate), 1);

    errors.forEach(e => {
      const el = document.createElement('div');
      el.className = 'error-bar-item';
      const color = errorColor(e.Error_Rate);
      el.innerHTML = `
        <span class="category-name">${e.Category}</span>
        <div class="bar-wrapper">
          <div class="bar-fill" style="width:0%;background:${color}" data-target="${(e.Error_Rate / maxErr * 100)}"></div>
        </div>
        <span class="error-value" style="color:${color}">${e.Error_Rate}%</span>
      `;
      container.appendChild(el);
    });

    // Animate bars
    setTimeout(() => {
      container.querySelectorAll('.bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.target + '%';
      });
    }, 200);
  }

  /* ── 9. Skill Matrix ─────────────────────────────────────────────────── */
  function renderSkillMatrix(data) {
    const skills = data.skill_matrix;
    const domains = Object.keys(skills);
    const categories = Object.keys(skills[domains[0]]).sort();

    // Collect all values
    let allVals = [];
    domains.forEach(d => categories.forEach(c => allVals.push(skills[d][c])));
    const minVal = Math.min(...allVals);
    const maxVal = Math.max(...allVals);

    // Head
    const thead = document.getElementById('skill-head');
    let headRow = '<tr><th class="row-header">Category</th>';
    domains.forEach(d => headRow += `<th>${d}</th>`);
    headRow += '</tr>';
    thead.innerHTML = headRow;

    // Body
    const tbody = document.getElementById('skill-body');
    categories.forEach(cat => {
      let row = `<tr><td class="cat-name">${cat}</td>`;
      domains.forEach(d => {
        const val = skills[d][cat];
        const bgColor = heatColor(val, minVal, maxVal);
        row += `<td style="background:${bgColor}">${val.toFixed(1)}</td>`;
      });
      row += '</tr>';
      tbody.innerHTML += row;
    });
  }

  /* ── 10. Vocabulary Stats ────────────────────────────────────────────── */
  function renderVocabulary(data) {
    const tbody = document.getElementById('vocab-body');
    const vocabData = data.vocabulary_stats;

    vocabData.forEach(v => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${v.Category}</td>
        <td>${v.Total_Words.toLocaleString()}</td>
        <td>${v.Unique_Words.toLocaleString()}</td>
        <td style="color:${v.TTR >= 0.1 ? COLORS.green : v.TTR >= 0.08 ? COLORS.orange : COLORS.red}">${v.TTR.toFixed(4)}</td>
        <td>${v.Avg_Word_Length}</td>
        <td>${(v.Stopword_Ratio * 100).toFixed(1)}%</td>
      `;
      tbody.appendChild(tr);
    });
  }

  /* ── 11. Feature Ablation ────────────────────────────────────────────── */
  function renderAblation(data) {
    const ablation = data.ablation_results;
    const labels = ablation.map(a => a.max_features.toLocaleString());
    const values = ablation.map(a => (a.F1 * 100).toFixed(2));

    new Chart(document.getElementById('ablation-chart'), {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Weighted F1 (%)',
          data: values,
          borderColor: COLORS.cyan,
          backgroundColor: COLORS.cyan + '18',
          borderWidth: 3,
          pointBackgroundColor: COLORS.cyan,
          pointRadius: 6,
          pointHoverRadius: 9,
          fill: true,
          tension: 0.3,
        }]
      },
      options: {
        ...CHART_DEFAULTS,
        plugins: {
          ...CHART_DEFAULTS.plugins,
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => `F1: ${ctx.raw}%`
            }
          }
        },
        scales: {
          x: {
            ...CHART_DEFAULTS.scales.x,
            title: { display: true, text: 'max_features', color: '#64748b', font: { family: 'JetBrains Mono' } }
          },
          y: {
            ...CHART_DEFAULTS.scales.y,
            title: { display: true, text: 'F1 Score (%)', color: '#64748b' },
            min: Math.max(0, Math.floor(Math.min(...values) / 5) * 5 - 5),
            max: Math.min(100, Math.ceil(Math.max(...values) / 5) * 5 + 5)
          }
        }
      }
    });
  }

  /* ── Scroll to Top Button ────────────────────────────────────────────── */
  function setupScrollTop() {
    const btn = document.getElementById('scroll-top-btn');
    window.addEventListener('scroll', () => {
      btn.classList.toggle('visible', window.scrollY > 500);
    });
    btn.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  /* ── Nav Highlight ───────────────────────────────────────────────────── */
  function setupNavHighlight() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-links a');

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          navLinks.forEach(link => {
            link.classList.toggle('active',
              link.getAttribute('href') === '#' + entry.target.id
            );
          });
        }
      });
    }, { rootMargin: '-30% 0px -60% 0px' });

    sections.forEach(s => observer.observe(s));
  }

})();
