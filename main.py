from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import json
import logging
from pathlib import Path
import tempfile
import platform
from scipy import stats  # Moved import to top

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Gene Expression Explorer API",
    description="Advanced API for gene expression analysis and visualization",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (dashboard HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Directories and cache config
# Determine the temporary directory based on the operating system
if platform.system() == "Windows":
    DATA_DIR = Path("./data")
else:
    DATA_DIR = Path(tempfile.gettempdir()) / "data"

# Create the directory if it doesn't exist, including parents
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_EXPIRY_HOURS = 24  # Fixed typo: 24v -> 24
MAX_CACHE_SIZE = 100

# Cache classes
class CacheEntry:
    def __init__(self, data: Any, expiry_hours: int = CACHE_EXPIRY_HOURS):
        self.data = data
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(hours=expiry_hours)

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

class EnhancedCache:
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                return None
            return entry.data
        return None

    def set(self, key: str, value: Any, expiry_hours: int = CACHE_EXPIRY_HOURS):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]
        self.cache[key] = CacheEntry(value, expiry_hours)

    def clear_expired(self):
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        for key in expired_keys:
            del self.cache[key]

# Initialize caches
dataset_cache = EnhancedCache()
model_cache = EnhancedCache()

# Pydantic models
class GeneQuery(BaseModel):
    gene_id: str
    dataset_id: str

    @validator("gene_id")
    def validate_gene_id(cls, v):
        if not v.strip():
            raise ValueError("Gene ID cannot be empty")
        return v.strip().upper()

    @validator("dataset_id")
    def validate_dataset_id(cls, v):
        if not v.strip().startswith("GSE"):
            raise ValueError("Dataset ID must start with GSE")
        return v.strip()

class MultiGeneQuery(BaseModel):
    gene_ids: List[str]
    dataset_id: str
    sample_count: Optional[int] = 20

    @validator("gene_ids")
    def validate_gene_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one gene ID is required")
        if len(v) > 50:
            raise ValueError("Maximum 50 genes allowed for heatmap")
        return [gene.strip().upper() for gene in v]

    @validator("dataset_id")
    def validate_dataset_id(cls, v):
        if not v.strip().startswith("GSE"):
            raise ValueError("Dataset ID must start with GSE")
        return v.strip()

class DatasetInfo(BaseModel):
    dataset_id: str
    title: str
    organism: str
    sample_count: int
    platform: Optional[str] = None
    summary: Optional[str] = None

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Gene Expression Explorer API</title>
        </head>
        <body>
            <h2>Welcome to Gene Expression Explorer API v2.0</h2>
            <p>This API provides features like gene expression visualization, ML prediction, and dashboard UI.</p>
            <p><a href="/dashboard">➡️ Go to Dashboard</a></p>
        </body>
    </html>
    """

# Serve dashboard HTML
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    try:
        dashboard_path = Path("static/dash.html")
        if not dashboard_path.exists():
            return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)
        return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"), status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {e}</h1>", status_code=500)

# Mock expression data generator
def generate_realistic_expression_data(gene_id: str, dataset_id: str) -> Dict[str, Any]:
    np.random.seed(hash(gene_id + dataset_id) % 2**32)
    base_expression = 8.0

    if any(marker in gene_id.upper() for marker in ["TP53", "BRCA", "MYC", "RAS"]):
        healthy = np.random.normal(base_expression, 1.5, 30)
        diseased = np.random.normal(base_expression + 4, 2.5, 30)
    elif any(marker in gene_id.upper() for marker in ["GAPDH", "ACTB", "TUBB"]):
        healthy = np.random.normal(base_expression + 2, 0.8, 30)
        diseased = np.random.normal(base_expression + 2.2, 1.0, 30)
    else:
        healthy = np.random.normal(base_expression, 2.0, 30)
        diseased = np.random.normal(base_expression + 2, 2.8, 30)

    return {"healthy": healthy.tolist(), "diseased": diseased.tolist()}

# Generate heatmap data for multiple samples
def generate_heatmap_data(gene_id: str, dataset_id: str, sample_count: int = 20) -> Dict[str, Any]:
    np.random.seed(hash(gene_id + dataset_id) % 2**32)
    
    # Generate sample names
    healthy_samples = [f"H_{i+1:02d}" for i in range(sample_count//2)]
    diseased_samples = [f"D_{i+1:02d}" for i in range(sample_count//2)]
    all_samples = healthy_samples + diseased_samples
    
    # Generate expression values with realistic patterns
    base_expression = 8.0
    if any(marker in gene_id.upper() for marker in ["TP53", "BRCA", "MYC", "RAS"]):
        healthy_expr = np.random.normal(base_expression, 1.5, len(healthy_samples))
        diseased_expr = np.random.normal(base_expression + 4, 2.5, len(diseased_samples))
    elif any(marker in gene_id.upper() for marker in ["GAPDH", "ACTB", "TUBB"]):
        healthy_expr = np.random.normal(base_expression + 2, 0.8, len(healthy_samples))
        diseased_expr = np.random.normal(base_expression + 2.2, 1.0, len(diseased_samples))
    else:
        healthy_expr = np.random.normal(base_expression, 2.0, len(healthy_samples))
        diseased_expr = np.random.normal(base_expression + 2, 2.8, len(diseased_samples))
    
    all_expressions = np.concatenate([healthy_expr, diseased_expr])
    
    return {
        "samples": all_samples,
        "expressions": all_expressions.tolist(),
        "conditions": ["Healthy"] * len(healthy_samples) + ["Diseased"] * len(diseased_samples)
    }

# Generate multi-gene heatmap data
def generate_multigene_heatmap_data(gene_ids: List[str], dataset_id: str, sample_count: int = 20) -> Dict[str, Any]:
    np.random.seed(hash(dataset_id + "".join(gene_ids)) % 2**32)
    
    # Generate sample names
    healthy_samples = [f"H_{i+1:02d}" for i in range(sample_count//2)]
    diseased_samples = [f"D_{i+1:02d}" for i in range(sample_count//2)]
    all_samples = healthy_samples + diseased_samples
    
    # Generate expression matrix
    expression_matrix = []
    for gene_id in gene_ids:
        data = generate_heatmap_data(gene_id, dataset_id, sample_count)
        expression_matrix.append(data["expressions"])
    
    return {
        "genes": [g.upper() for g in gene_ids],
        "samples": all_samples,
        "expression_matrix": expression_matrix,
        "conditions": ["Healthy"] * len(healthy_samples) + ["Diseased"] * len(diseased_samples)
    }

# Benjamini-Hochberg FDR calculation
def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """
    Apply the Benjamini-Hochberg FDR correction to a list of p-values.
    Returns the adjusted p-values.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and keep track of their original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]
    
    # Compute adjusted p-values
    adjusted_pvals = np.zeros(n)
    cumulative_min = np.inf
    for i in range(n-1, -1, -1):
        rank = i + 1
        adjusted = sorted_pvals[i] * n / rank
        cumulative_min = min(cumulative_min, adjusted)
        adjusted_pvals[i] = min(cumulative_min, 1.0)  # Cap at 1.0
    
    # Restore original order
    final_adjusted = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        final_adjusted[idx] = adjusted_pvals[i]
    
    return final_adjusted.tolist()

# Stats calculator
def calculate_statistics(healthy: List[float], diseased: List[float]) -> Dict[str, Any]:
    healthy = np.array(healthy)
    diseased = np.array(diseased)
    t_stat, p_value = stats.ttest_ind(healthy, diseased, equal_var=False)  # Welch's t-test

    # Since this is a single gene test, we only have one p-value
    # Apply Benjamini-Hochberg as if it's a single test (for consistency with dashboard expectation)
    p_values = [p_value]
    adjusted_pvals = benjamini_hochberg(p_values)

    return {
        "healthy_mean": float(np.mean(healthy)),
        "healthy_std": float(np.std(healthy)),
        "diseased_mean": float(np.mean(diseased)),
        "diseased_std": float(np.std(diseased)),
        "fold_change": float(np.mean(diseased) / np.mean(healthy)) if np.mean(healthy) != 0 else float("inf"),
        "log2_fold_change": float(np.log2(np.mean(diseased) / np.mean(healthy))) if np.mean(healthy) > 0 else float("inf"),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "adjusted_p_value": adjusted_pvals[0],  # FDR-adjusted p-value
        "benjamini_hochberg_fdr": adjusted_pvals,  # List of adjusted p-values (matches dashboard expectation)
        "significant": bool(p_value < 0.05)
    }

# Visualization endpoint
@app.get("/visualization/{dataset_id}/{gene_id}")
async def visualize_gene_expression(dataset_id: str, gene_id: str, plot_type: str = "boxplot"):
    try:
        cache_key = f"{dataset_id}_{gene_id}_{plot_type}"
        cached_result = dataset_cache.get(cache_key)
        if cached_result:
            return cached_result

        if plot_type == "heatmap":
            # Generate heatmap for single gene across samples
            heatmap_data = generate_heatmap_data(gene_id, dataset_id)
            
            # Create heatmap with plotly
            fig = px.imshow(
                [heatmap_data["expressions"]], 
                x=heatmap_data["samples"],
                y=[gene_id.upper()],
                color_continuous_scale="RdYlBu_r",
                aspect="auto",
                title=f"Gene Expression Heatmap: {gene_id.upper()} ({dataset_id})"
            )
            
            # Add condition annotations
            fig.add_annotation(
                x=4.5, y=-0.7, text="Healthy", showarrow=False, 
                font=dict(size=12, color="green")
            )
            fig.add_annotation(
                x=14.5, y=-0.7, text="Diseased", showarrow=False,
                font=dict(size=12, color="red")
            )
            
            plot_json = json.loads(fig.to_json())
            
            response = {
                "gene_id": gene_id.upper(),
                "dataset_id": dataset_id,
                "plot_type": plot_type,
                "plot_data": plot_json,
                "heatmap_data": heatmap_data
            }
            
        else:
            data = generate_realistic_expression_data(gene_id, dataset_id)
            stats = calculate_statistics(data["healthy"], data["diseased"])

            df = pd.DataFrame({
                "Expression": data["healthy"] + data["diseased"],
                "Condition": ["Healthy"] * len(data["healthy"]) + ["Diseased"] * len(data["diseased"])
            })

            if plot_type == "violin":
                fig = px.violin(df, y="Expression", x="Condition", box=True, points="all", color="Condition")
            elif plot_type == "histogram":
                fig = px.histogram(df, x="Expression", color="Condition", barmode="overlay", nbins=20, opacity=0.75)
            else:
                fig = px.box(df, y="Expression", x="Condition", color="Condition", points="all")

            plot_json = json.loads(fig.to_json())

            response = {
                "gene_id": gene_id.upper(),
                "dataset_id": dataset_id,
                "plot_type": plot_type,
                "plot_data": plot_json,
                "statistics": stats
            }

        dataset_cache.set(cache_key, response)
        return response

    except Exception as e:
        logger.error(f"Error generating visualization for {gene_id} in {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

# Multi-gene heatmap endpoint
@app.post("/heatmap/multigene")
async def create_multigene_heatmap(query: MultiGeneQuery):
    try:
        cache_key = f"multigene_{query.dataset_id}_{'_'.join(query.gene_ids)}_{query.sample_count}"
        cached_result = dataset_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Generate multi-gene heatmap data
        heatmap_data = generate_multigene_heatmap_data(query.gene_ids, query.dataset_id, query.sample_count)
        
        # Create clustered heatmap
        fig = px.imshow(
            heatmap_data["expression_matrix"],
            x=heatmap_data["samples"],
            y=heatmap_data["genes"],
            color_continuous_scale="RdYlBu_r",
            aspect="auto",
            title=f"Multi-Gene Expression Heatmap ({query.dataset_id})",
            labels=dict(x="Samples", y="Genes", color="Expression Level")
        )
        
        # Add condition annotations
        fig.add_annotation(
            x=query.sample_count//4 - 0.5, y=-0.7, text="Healthy", showarrow=False,
            font=dict(size=12, color="green")
        )
        fig.add_annotation(
            x=3*query.sample_count//4 - 0.5, y=-0.7, text="Diseased", showarrow=False,
            font=dict(size=12, color="red")
        )
        
        # Add vertical line to separate conditions
        fig.add_vline(x=query.sample_count//2 - 0.5, line_dash="dash", line_color="black", opacity=0.5)
        
        plot_json = json.loads(fig.to_json())
        
        response = {
            "gene_ids": heatmap_data["genes"],
            "dataset_id": query.dataset_id,
            "sample_count": query.sample_count,
            "plot_type": "multi_gene_heatmap",
            "plot_data": plot_json,
            "heatmap_data": heatmap_data
        }
        
        dataset_cache.set(cache_key, response)
        return response
        
    except Exception as e:
        logger.error(f"Error generating multi-gene heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-gene heatmap error: {str(e)}")

# Correlation heatmap endpoint
@app.get("/heatmap/correlation/{dataset_id}")
async def create_correlation_heatmap(dataset_id: str, gene_ids: str = "TP53,BRCA1,MYC,RAS,GAPDH"):
    try:
        gene_list = [g.strip().upper() for g in gene_ids.split(",")]
        cache_key = f"correlation_{dataset_id}_{'_'.join(gene_list)}"
        cached_result = dataset_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Generate expression data for all genes
        expression_data = {}
        for gene in gene_list:
            data = generate_realistic_expression_data(gene, dataset_id)
            expression_data[gene] = data["healthy"] + data["diseased"]
        
        # Create correlation matrix
        df = pd.DataFrame(expression_data)
        correlation_matrix = df.corr().values
        
        # Create correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            x=gene_list,
            y=gene_list,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            aspect="auto",
            title=f"Gene Expression Correlation Matrix ({dataset_id})",
            labels=dict(color="Correlation")
        )
        
        # Add correlation values as text
        for i in range(len(gene_list)):
            for j in range(len(gene_list)):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{correlation_matrix[i,j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(correlation_matrix[i,j]) > 0.5 else "black")
                )
        
        plot_json = json.loads(fig.to_json())
        
        response = {
            "gene_ids": gene_list,
            "dataset_id": dataset_id,
            "plot_type": "correlation_heatmap",
            "plot_data": plot_json,
            "correlation_matrix": correlation_matrix.tolist()
        }
        
        dataset_cache.set(cache_key, response)
        return response
        
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Correlation heatmap error: {str(e)}")

# Additional utility endpoints
@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return {
        "dataset_cache_size": len(dataset_cache.cache),
        "model_cache_size": len(model_cache.cache),
        "dataset_cache_max": dataset_cache.max_size,
        "model_cache_max": model_cache.max_size
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    dataset_cache.cache.clear()
    model_cache.cache.clear()
    dataset_cache.clear_expired()
    model_cache.clear_expired()
    return {"message": "All caches cleared successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "features": [
            "gene_expression_visualization",
            "multi_gene_heatmaps",
            "correlation_analysis",
            "statistical_analysis",
            "caching_system"
        ]
    }

# ML Model Training and Prediction Endpoints
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Generate mock training data for ML model
def generate_training_data(num_samples=100):
    np.random.seed(42)
    # Simulate gene expression data for 5 genes
    num_features = 5
    healthy_samples = np.random.normal(loc=8.0, scale=1.5, size=(num_samples // 2, num_features))
    diseased_samples = np.random.normal(loc=12.0, scale=2.5, size=(num_samples // 2, num_features))
    X = np.vstack((healthy_samples, diseased_samples))
    y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))  # 0: Healthy, 1: Diseased
    return X, y

# Train a simple ML model
def train_model():
    X, y = generate_training_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# Train the model at startup and store in model_cache
model, scaler = train_model()
model_cache.set("prediction_model", {"model": model, "scaler": scaler})

# Endpoint to fetch sample IDs
@app.get("/api/sample-ids")
async def get_sample_ids(dataset: str):
    try:
        if not dataset.startswith("GSE"):
            raise HTTPException(status_code=400, detail="Dataset must start with GSE")

        # Generate mock sample IDs (replace with real data if available)
        sample_ids = [f"GSM{i:06d}" for i in range(500001, 500021)]  # e.g., GSM500001 to GSM500020
        return {"sample_ids": sample_ids}

    except Exception as e:
        logger.error(f"Error fetching sample IDs for {dataset}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching sample IDs: {str(e)}")

# Endpoint to predict sample condition
@app.get("/api/predict-sample")
async def predict_sample(dataset: str, sample_id: str):
    try:
        # Validate dataset and sample_id
        if not dataset.startswith("GSE"):
            raise HTTPException(status_code=400, detail="Dataset must start with GSE")
        if not sample_id:
            raise HTTPException(status_code=400, detail="Sample ID cannot be empty")

        # Fetch the model and scaler from cache
        cached_model = model_cache.get("prediction_model")
        if not cached_model:
            raise HTTPException(status_code=500, detail="ML model not available")

        model = cached_model["model"]
        scaler = cached_model["scaler"]

        # Generate mock gene expression data for this sample (replace with real data if available)
        np.random.seed(hash(sample_id) % 2**32)
        gene_expression = np.random.normal(loc=10.0, scale=2.0, size=(1, 5))  # 5 features

        # Scale the data
        gene_expression_scaled = scaler.transform(gene_expression)

        # Make prediction
        prediction = model.predict(gene_expression_scaled)[0]
        probabilities = model.predict_proba(gene_expression_scaled)[0]

        # Format response
        result = {
            "sample_id": sample_id,
            "prediction": "Diseased" if prediction == 1 else "Healthy",
            "probability_healthy": float(probabilities[0]),
            "probability_diseased": float(probabilities[1])
        }
        return result

    except Exception as e:
        logger.error(f"Error predicting sample {sample_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Keep the Uvicorn block commented out for Vercel deployment
#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)