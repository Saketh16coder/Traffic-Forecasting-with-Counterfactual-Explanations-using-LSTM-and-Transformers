// API Configuration
const API_URL = 'http://localhost:5000';

// State
let featureChart = null;
let attentionChart = null;

// Chart.js global dark theme
Chart.defaults.color = '#94A3B8';
Chart.defaults.borderColor = '#1E293B';
Chart.defaults.font.family = "'DM Sans', sans-serif";

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    setupFileUpload();
    setDefaultValues();
});

// Set default input values based on dataset ranges
function setDefaultValues() {
    const now = new Date();
    document.getElementById('hour').value = now.getHours();
    document.getElementById('day').value = now.getDay() === 0 ? 6 : now.getDay() - 1;
    document.getElementById('speed').value = 58;
    document.getElementById('volume').value = 20;

    // Try to fetch actual data stats to update placeholders
    fetchDataStats();
}

// Fetch data statistics to show proper ranges
async function fetchDataStats() {
    try {
        const response = await fetch(`${API_URL}/data-stats`);
        if (response.ok) {
            const stats = await response.json();
            document.getElementById('speed').placeholder = `${stats.speed.mean} (${stats.speed.min}-${stats.speed.max})`;
            document.getElementById('volume').placeholder = `${stats.volume.mean} (${stats.volume.min}-${stats.volume.max})`;
        }
    } catch (error) {
        // Silently ignore - defaults are fine
    }
}

// Setup file upload drag and drop
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileUpload');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.style.borderColor = '#00E5CC';
            uploadArea.style.background = 'rgba(0, 229, 204, 0.04)';
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.style.borderColor = '';
            uploadArea.style.background = '';
        });
    });

    uploadArea.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            uploadFile();
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            uploadFile();
        }
    });
}

// Check API status
async function checkStatus() {
    try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();

        updateStatusIndicator('data', data.data_uploaded);
        updateStatusIndicator('model', data.model_trained);

        if (data.data_info) {
            showStatus('uploadStatus', `Data loaded: ${data.data_info.rows} rows`, 'success');
        }
    } catch (error) {
        console.log('API not available yet');
    }
}

// Update status indicator
function updateStatusIndicator(type, isActive) {
    const dot = document.getElementById(`${type}Status`);
    const text = document.getElementById(`${type}StatusText`);

    if (isActive) {
        dot.classList.add('active');
        text.textContent = type === 'data' ? 'Ready' : 'Trained';
    } else {
        dot.classList.remove('active');
        text.textContent = type === 'data' ? 'Offline' : 'Untrained';
    }
}

// Show status message
function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = `toast ${type}`;
}

// Upload file
async function uploadFile() {
    const fileInput = document.getElementById('fileUpload');

    if (!fileInput.files.length) {
        showStatus('uploadStatus', 'Please select a file first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    showStatus('uploadStatus', 'Uploading...', 'info');

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showStatus('uploadStatus', `${data.message} (${data.rows} rows)`, 'success');
            updateStatusIndicator('data', true);
        } else {
            showStatus('uploadStatus', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('uploadStatus', `Error: ${error.message}`, 'error');
    }
}

// Generate sample data
async function generateSampleData() {
    showStatus('uploadStatus', 'Generating sample data...', 'info');

    try {
        const response = await fetch(`${API_URL}/generate-sample`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hours: 168 })
        });

        const data = await response.json();

        if (response.ok) {
            showStatus('uploadStatus', data.message, 'success');
            updateStatusIndicator('data', true);
        } else {
            showStatus('uploadStatus', `Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus('uploadStatus', `Error: ${error.message}`, 'error');
    }
}

// Train model with real progress polling
async function trainModel() {
    const trainBtn = document.getElementById('trainBtn');
    const progressContainer = document.getElementById('trainingProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    trainBtn.disabled = true;
    trainBtn.innerHTML = '<span class="loading"></span> Training...';
    progressContainer.style.display = 'flex';
    progressFill.style.width = '0%';
    progressText.textContent = 'Starting...';

    try {
        // Kick off training (returns immediately)
        const startResp = await fetch(`${API_URL}/train`, { method: 'POST' });
        const startData = await startResp.json();

        if (!startResp.ok) {
            showStatus('trainStatus', `Error: ${startData.error}`, 'error');
            return;
        }

        // Poll /train/status for real epoch-level progress
        const result = await new Promise((resolve, reject) => {
            const poll = setInterval(async () => {
                try {
                    const resp = await fetch(`${API_URL}/train/status`);
                    const data = await resp.json();

                    if (data.status === 'training') {
                        const pct = Math.max(data.progress, 1);
                        progressFill.style.width = `${pct}%`;

                        // Build a rich status line
                        let label = `Epoch ${data.epoch}/${data.total_epochs}`;
                        if (data.train_loss !== null) {
                            label += `  \u2022  loss: ${data.train_loss.toFixed(4)}`;
                        }
                        if (data.val_loss !== null) {
                            label += `  \u2022  val: ${data.val_loss.toFixed(4)}`;
                        }
                        progressText.textContent = label;
                    } else if (data.status === 'success') {
                        clearInterval(poll);
                        progressFill.style.width = '100%';
                        progressText.textContent = 'Complete!';
                        resolve(data);
                    } else if (data.status === 'error') {
                        clearInterval(poll);
                        reject(new Error(data.error));
                    }
                } catch (e) {
                    // Server busy, keep polling
                }
            }, 800);
        });

        let msg = `Model trained — Loss: ${result.final_loss.toFixed(4)} | ${result.training_samples} samples`;
        if (result.metrics) {
            const m = result.metrics;
            msg += ` | MAE: ${m.mae} mph | R\u00b2: ${m.r2}`;
            msg += ` | Accuracy (\u00b13mph): ${m.accuracy_within_3mph}%`;
            msg += ` | Accuracy (\u00b15mph): ${m.accuracy_within_5mph}%`;
        }
        showStatus('trainStatus', msg, 'success');
        updateStatusIndicator('model', true);

    } catch (error) {
        showStatus('trainStatus', `Error: ${error.message}`, 'error');
    } finally {
        trainBtn.disabled = false;
        trainBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Train Model`;
        setTimeout(() => {
            progressContainer.style.display = 'none';
            progressFill.style.width = '0%';
        }, 2000);
    }
}

// Make prediction
async function predict() {
    const speed = parseFloat(document.getElementById('speed').value);
    const volume = parseFloat(document.getElementById('volume').value);
    const hour = parseInt(document.getElementById('hour').value);
    const day = parseInt(document.getElementById('day').value);

    // Validate inputs
    if (isNaN(speed) || isNaN(volume) || isNaN(hour) || isNaN(day)) {
        alert('Please fill in all fields with valid numbers');
        return;
    }

    const predictBtn = document.getElementById('predictBtn');
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="loading"></span> Predicting...';

    // Create 12 hours of historical data (simulated with small perturbations)
    const historicalData = [];
    for (let i = 11; i >= 0; i--) {
        const pastHour = (hour - i + 24) % 24;
        historicalData.push([
            Math.max(0, speed + (Math.random() - 0.5) * 5),
            Math.max(0, volume + (Math.random() - 0.5) * 6),
            pastHour,
            day
        ]);
    }

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: historicalData })
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Predict Traffic Speed`;
    }
}

// Display prediction results
function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    // Small delay for animation
    requestAnimationFrame(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    // Update prediction value
    const prediction = data.prediction;
    document.getElementById('predictionValue').textContent = prediction;

    // Update condition badge
    const badge = document.getElementById('conditionBadge');
    if (prediction >= 55) {
        badge.textContent = 'Free Flowing';
        badge.className = 'badge good';
    } else if (prediction >= 40) {
        badge.textContent = 'Moderate Traffic';
        badge.className = 'badge moderate';
    } else {
        badge.textContent = 'Congested';
        badge.className = 'badge congested';
    }

    // Update ring border color to match condition
    const ring = document.querySelector('.results__ring');
    if (prediction >= 55) {
        ring.style.borderColor = '#34D399';
        ring.style.boxShadow = '0 0 40px rgba(52,211,153,0.3), inset 0 0 30px rgba(52,211,153,0.04)';
    } else if (prediction >= 40) {
        ring.style.borderColor = '#FFB020';
        ring.style.boxShadow = '0 0 40px rgba(255,176,32,0.3), inset 0 0 30px rgba(255,176,32,0.04)';
    } else {
        ring.style.borderColor = '#FF4D6A';
        ring.style.boxShadow = '0 0 40px rgba(255,77,106,0.3), inset 0 0 30px rgba(255,77,106,0.04)';
    }

    // Populate input summary cards
    const summary = data.explanation.input_summary;
    if (summary) {
        const dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
        document.getElementById('summarySpeed').textContent = summary.avg_speed;
        document.getElementById('summaryVolume').textContent = summary.avg_volume;
        document.getElementById('summaryHour').textContent = summary.current_hour;
        document.getElementById('summaryHourLabel').textContent =
            summary.current_hour >= 12 ? `${summary.current_hour === 12 ? 12 : summary.current_hour - 12} PM` : `${summary.current_hour === 0 ? 12 : summary.current_hour} AM`;
        document.getElementById('summaryDay').textContent = dayNames[summary.day_of_week] || summary.day_of_week;
        document.getElementById('summaryDayLabel').textContent = summary.day_of_week >= 5 ? 'Weekend' : 'Weekday';
    }

    // Update explanation
    document.getElementById('explanationText').textContent = data.explanation.explanation;

    // Draw feature importance chart
    drawFeatureChart(data.explanation.feature_importance);

    // Draw temporal attention chart
    const attnData = data.temporal_attention || (data.explanation && data.explanation.temporal_attention);
    drawAttentionChart(attnData);

    // Display counterfactual scenarios
    displayCounterfactuals(data.explanation.counterfactual);
}

// Draw feature importance chart
function drawFeatureChart(importance) {
    const ctx = document.getElementById('featureChart').getContext('2d');

    if (featureChart) featureChart.destroy();

    const labels = {
        'speed': 'Upstream Speed',
        'volume': 'Traffic Volume',
        'hour': 'Time of Day',
        'day_of_week': 'Day of Week'
    };

    const chartLabels = Object.keys(importance).map(k => labels[k] || k);
    const chartData = Object.values(importance);

    const colors = [
        { bg: 'rgba(0, 229, 204, 0.7)',  border: '#00E5CC' },
        { bg: 'rgba(52, 211, 153, 0.7)',  border: '#34D399' },
        { bg: 'rgba(255, 176, 32, 0.7)',  border: '#FFB020' },
        { bg: 'rgba(167, 139, 250, 0.7)', border: '#A78BFA' },
    ];

    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Impact (%)',
                data: chartData,
                backgroundColor: colors.map(c => c.bg),
                borderColor: colors.map(c => c.border),
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1A2035',
                    borderColor: '#1E293B',
                    borderWidth: 1,
                    titleColor: '#F1F5F9',
                    bodyColor: '#94A3B8',
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: ctx => `Impact: ${ctx.raw.toFixed(1)}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: {
                        callback: v => v + '%',
                        font: { size: 11 }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } }
                }
            }
        }
    });
}

// Draw temporal attention chart
function drawAttentionChart(attnData) {
    const container = document.getElementById('attentionChartContainer');
    const descriptionEl = document.getElementById('attentionDescription');

    if (!attnData || !attnData.weights) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';

    if (attnData.description) {
        descriptionEl.textContent = attnData.description;
        descriptionEl.style.display = 'block';
    } else {
        descriptionEl.style.display = 'none';
    }

    const ctx = document.getElementById('attentionChart').getContext('2d');

    if (attentionChart) attentionChart.destroy();

    const chartLabels = attnData.labels
        ? attnData.labels.map(l => {
            const num = parseInt(l.replace('t-', ''));
            return num === 0 ? 'Current' : `${num}h ago`;
          })
        : attnData.weights.map((_, i) => {
            const hoursAgo = attnData.weights.length - 1 - i;
            return hoursAgo === 0 ? 'Current' : `${hoursAgo}h ago`;
          });

    const weights = attnData.weights;
    const maxWeight = Math.max(...weights);

    const backgroundColors = weights.map(w => {
        const intensity = Math.max(0.15, w / (maxWeight + 1e-8));
        return `rgba(167, 139, 250, ${intensity * 0.85})`;
    });

    attentionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Attention Weight',
                data: weights.map(w => parseFloat((w * 100).toFixed(1))),
                backgroundColor: backgroundColors,
                borderColor: 'rgba(167, 139, 250, 0.6)',
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1A2035',
                    borderColor: '#1E293B',
                    borderWidth: 1,
                    titleColor: '#F1F5F9',
                    bodyColor: '#94A3B8',
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: ctx => `Attention: ${ctx.raw.toFixed(1)}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    title: {
                        display: true,
                        text: 'Attention Weight (%)',
                        color: '#64748B',
                        font: { size: 11 }
                    },
                    ticks: {
                        callback: v => v + '%',
                        font: { size: 11 }
                    }
                },
                x: {
                    grid: { display: false },
                    title: {
                        display: true,
                        text: 'Time Step',
                        color: '#64748B',
                        font: { size: 11 }
                    },
                    ticks: { font: { size: 11 } }
                }
            }
        }
    });
}

// Display counterfactual scenarios
function displayCounterfactuals(counterfactual) {
    const container = document.getElementById('counterfactualCards');
    container.innerHTML = '';

    if (!counterfactual || !counterfactual.scenarios) return;

    counterfactual.scenarios.forEach(scenario => {
        const card = document.createElement('div');
        card.className = 'scenario-card';

        const changeClass = scenario.change >= 0 ? 'positive' : 'negative';
        const changeSymbol = scenario.change >= 0 ? '+' : '';

        card.innerHTML = `
            <h4>${scenario.description}</h4>
            <div class="scenario-change ${changeClass}">
                ${changeSymbol}${scenario.change.toFixed(1)} mph
            </div>
            <div class="scenario-description">
                ${scenario.original_prediction} &rarr; ${scenario.new_prediction} mph
            </div>
        `;

        container.appendChild(card);
    });
}
