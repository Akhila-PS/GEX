// API Base URL
const API_BASE = "http://localhost:8000";

// Tab switching functionality
function switchTab(tabName) {
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => content.classList.remove('active'));
    
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    const targetTabContent = document.getElementById(tabName);
    if (targetTabContent) {
        targetTabContent.classList.add('active');
    }
    
    const targetTab = document.querySelector(`button[onclick="switchTab('${tabName}')"]`);
    if (targetTab) {
        targetTab.classList.add('active');
    }
    
    hideMessages();
}

// Utility functions
function showLoading() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    hideMessages();
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    document.getElementById('success').style.display = 'none';
}

function showSuccess(message) {
    const successDiv = document.getElementById('success');
    successDiv.textContent = message;
    successDiv.style.display = 'block';
    document.getElementById('error').style.display = 'none';
}

function hideMessages() {
    document.getElementById('error').style.display = 'none';
    document.getElementById('success').style.display = 'none';
}

// Single gene analysis
async function analyzeSingleGene() {
    const geneId = document.getElementById('geneId').value.trim();
    const datasetId = document.getElementById('datasetId').value.trim();
    const plotType = document.getElementById('plotType').value;

    if (!geneId || !datasetId) {
        showError('Please enter both Gene ID and Dataset ID');
        return;
    }

    if (!datasetId.startsWith('GSE')) {
        showError('Dataset ID must start with GSE');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${API_BASE}/visualization/${datasetId}/${geneId}?plot_type=${plotType}`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        hideLoading();
        displayResults(data);
        showSuccess(`Successfully analyzed ${geneId} in ${datasetId}`);

    } catch (error) {
        hideLoading();
        showError(`Error analyzing gene: ${error.message}`);
        console.error('Analysis error:', error);
    }
}

// Multi-gene heatmap analysis
async function analyzeMultiGene() {
    const geneIds = document.getElementById('multiGeneIds').value.trim();
    const datasetId = document.getElementById('multiDatasetId').value.trim();
    const sampleCount = parseInt(document.getElementById('sampleCount').value);

    if (!geneIds || !datasetId) {
        showError('Please enter both Gene IDs and Dataset ID');
        return;
    }

    const geneList = geneIds.split(',').map(g => g.trim()).filter(g => g);
    
    if (geneList.length === 0) {
        showError('Please enter at least one gene ID');
        return;
    }

    if (geneList.length > 50) {
        showError('Maximum 50 genes allowed');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${API_BASE}/heatmap/multigene`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                gene_ids: geneList,
                dataset_id: datasetId,
                sample_count: sampleCount
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        hideLoading();
        displayResults(data);
        showSuccess(`Successfully generated heatmap for ${geneList.length} genes`);

    } catch (error) {
        hideLoading();
        showError(`Error generating heatmap: ${error.message}`);
        console.error('Heatmap error:', error);
    }
}

// Correlation analysis
async function analyzeCorrelation() {
    const datasetId = document.getElementById('corrDatasetId').value.trim();
    const geneIds = document.getElementById('corrGeneIds').value.trim();

    if (!datasetId || !geneIds) {
        showError('Please enter both Dataset ID and Gene IDs');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${API_BASE}/heatmap/correlation/${datasetId}?gene_ids=${encodeURIComponent(geneIds)}`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        hideLoading();
        displayResults(data);
        showSuccess('Successfully generated correlation matrix');

    } catch (error) {
        hideLoading();
        showError(`Error generating correlation: ${error.message}`);
        console.error('Correlation error:', error);
    }
}

// Display results
function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const chartDiv = document.getElementById('chart');
    const chartTitle = document.getElementById('chartTitle');
    const statsContainer = document.getElementById('statsContainer');
    const statsDiv = document.getElementById('stats');

    if (data.gene_id) {
        chartTitle.textContent = `${data.gene_id} Expression Analysis (${data.dataset_id})`;
    } else if (data.gene_ids) {
        chartTitle.textContent = `Multi-Gene Analysis (${data.dataset_id})`;
    }

    if (data.plot_data) {
        Plotly.newPlot(chartDiv, data.plot_data.data, data.plot_data.layout, {responsive: true});
    }

    if (data.statistics) {
        displayStatistics(data.statistics, statsDiv);
        statsContainer.style.display = 'block';
    } else {
        statsContainer.style.display = 'none';
    }

    resultsDiv.style.display = 'block';
}

// Display statistics
function displayStatistics(stats, container) {
    container.innerHTML = '';
    
    const statItems = [
        { label: 'Healthy Mean', value: stats.healthy_mean?.toFixed(3) },
        { label: 'Diseased Mean', value: stats.diseased_mean?.toFixed(3) },
        { label: 'Fold Change', value: stats.fold_change?.toFixed(3) },
        { label: 'Log2 FC', value: stats.log2_fold_change?.toFixed(3) },
        { label: 'P-Value (Welch\'s t-test)', value: stats.p_value?.toExponential(3) },
        { label: 'P-Value Adjusted (FDR)', value: stats.adjusted_p_value?.toExponential(3) },
        { label: 'Benjamini-Hochberg FDR', value: stats.benjamini_hochberg_fdr?.map(p => p.toExponential(3)).join(', ') },
        { label: 'Significant', value: stats.significant ? 'Yes' : 'No' }
    ];

    statItems.forEach(item => {
        if (item.value !== undefined) {
            const statDiv = document.createElement('div');
            statDiv.className = 'stat-item';
            statDiv.innerHTML = `
                <div class="stat-value">${item.value}</div>
                <div class="stat-label">${item.label}</div>
            `;
            container.appendChild(statDiv);
        }
    });
}

// Cache and system functions
async function getCacheStats() {
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/cache/stats`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        hideLoading();
        displaySystemInfo(data);
        showSuccess('Cache stats retrieved successfully');
    } catch (error) {
        hideLoading();
        showError(`Error getting cache stats: ${error.message}`);
        console.error('Cache stats error:', error);
    }
}

async function clearCache() {
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/cache/clear`, { method: 'POST' });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        hideLoading();
        showSuccess(data.message || 'Cache cleared successfully');
    } catch (error) {
        hideLoading();
        showError(`Error clearing cache: ${error.message}`);
        console.error('Clear cache error:', error);
    }
}

async function healthCheck() {
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        hideLoading();
        displaySystemInfo(data);
        showSuccess(`System is ${data.status}`);
    } catch (error) {
        hideLoading();
        showError(`Error checking health: ${error.message}`);
        console.error('Health check error:', error);
    }
}

function displaySystemInfo(data) {
    const systemInfo = document.getElementById('systemInfo');
    const systemDetails = document.getElementById('systemDetails');
    
    systemDetails.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    systemInfo.style.display = 'block';
}

// Prediction AI Functions
async function fetchSampleIds() {
    const datasetId = document.getElementById('predictDatasetId').value.trim();
    const chatMessages = document.getElementById('chatMessages');

    if (!datasetId) {
        showError('Please enter a Dataset ID');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot';
        messageDiv.textContent = 'Please enter a Dataset ID.';
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return;
    }

    if (!datasetId.startsWith('GSE')) {
        showError('Dataset ID must start with GSE');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot';
        messageDiv.textContent = 'Dataset ID must start with GSE. Try something like GSE19804.';
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return;
    }

    showLoading();

    try {
        const url = `${API_BASE}/api/sample-ids?dataset=${encodeURIComponent(datasetId)}`;
        console.log('Fetching sample IDs from:', url);
        const response = await fetch(url);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Sample IDs response:', data);

        hideLoading();

        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot';
        messageDiv.textContent = `Here are some valid sample IDs: ${data.sample_ids.slice(0, 5).join(', ')}${data.sample_ids.length > 5 ? ` and ${data.sample_ids.length - 5} more` : ''}`;
        chatMessages.appendChild(messageDiv);

        chatMessages.scrollTop = chatMessages.scrollHeight;
        showSuccess('Sample IDs fetched successfully');
    } catch (error) {
        hideLoading();
        showError(`Error fetching sample IDs: ${error.message}`);
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot';
        messageDiv.textContent = `Error: ${error.message}`;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        console.error('Fetch sample IDs error:', error);
    }
}

async function predictSample() {
    const datasetId = document.getElementById('predictDatasetId').value.trim();
    const sampleId = document.getElementById('sampleId').value.trim();
    const chatMessages = document.getElementById('chatMessages');

    if (!datasetId || !sampleId) {
        showError('Please enter both Dataset ID and Sample ID');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot';
        messageDiv.textContent = 'I need both a Dataset ID and a Sample ID to make a prediction.';
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return;
    }

    if (!datasetId.startsWith('GSE')) {
        showError('Dataset ID must start with GSE');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot';
        messageDiv.textContent = 'Dataset ID must start with GSE. Try something like GSE19804.';
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return;
    }

    if (!sampleId.startsWith('GSM')) {
        showError('Sample ID must start with GSM');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot';
        messageDiv.textContent = 'Sample ID must start with GSM. Try one of the IDs I shared!';
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return;
    }

    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'chat-message user';
    userMessageDiv.textContent = sampleId;
    chatMessages.appendChild(userMessageDiv);

    showLoading();

    try {
        const url = `${API_BASE}/api/predict-sample?dataset=${encodeURIComponent(datasetId)}&sample_id=${encodeURIComponent(sampleId)}`;
        console.log('Fetching prediction from:', url);
        const response = await fetch(url);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Prediction response:', data);

        hideLoading();

        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot';
        messageDiv.textContent = `Sample ${data.sample_id}: ${data.prediction} (Probability Healthy: ${data.probability_healthy.toFixed(2)}, Diseased: ${data.probability_diseased.toFixed(2)})`;
        chatMessages.appendChild(messageDiv);

        chatMessages.scrollTop = chatMessages.scrollHeight;
        showSuccess('Prediction successful');
    } catch (error) {
        hideLoading();
        showError(`Error making prediction: ${error.message}`);
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message bot';
        messageDiv.textContent = `Error: ${error.message}`;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        console.error('Predict sample error:', error);
    }

    document.getElementById('sampleId').value = '';
}

function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '<div class="chat-message bot">Hi! I can predict if a sample is healthy or diseased. Enter a dataset ID and click "Fetch Sample IDs" to get started.</div>';
    showSuccess('Chat cleared');
}

// Quick action functions
function loadSampleData() {
    document.getElementById('geneId').value = 'TP53';
    document.getElementById('datasetId').value = 'GSE123456';
    document.getElementById('plotType').value = 'boxplot';
    document.getElementById('predictDatasetId').value = 'GSE19804';
    showSuccess('Sample data loaded');
}

function downloadResults() {
    const resultsDiv = document.getElementById('results');
    if (resultsDiv.style.display === 'none') {
        showError('No results to download');
        return;
    }
    showSuccess('Download functionality would be implemented here');
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Gene Expression Explorer Dashboard loaded');
    loadSampleData();
});