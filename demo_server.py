import os
import glob
import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Ensure we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from aggregator.aggregator import evaluate_text
    from benchmark.run_benchmark import evaluate_on_benchmark
except ImportError as e:
    print(f"Error importing safeguards: {e}")
    print("Make sure you are running this from the project root.")
    sys.exit(1)

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")

class AnalyzeRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    try:
        result = evaluate_text(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmarks/latest")
async def get_latest_benchmark():
    # Find the most recent benchmark results file
    benchmark_dir = Path("benchmark")
    files = list(benchmark_dir.glob("benchmark_results_*.json"))
    
    if not files:
        return {"error": "No benchmark results found"}
    
    # Sort by modification time, newest first
    latest_file = max(files, key=os.path.getmtime)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    return data

@app.get("/api/run-benchmark")
async def run_benchmark_endpoint(benchmark: str = "HarmBench", limit: int = 5):
    """Run a quick benchmark on demand (limited examples for speed)"""
    try:
        # Use the function from run_benchmark.py
        result = evaluate_on_benchmark(
            benchmark_name=benchmark,
            limit=limit,
            threshold=0.5,
            verbose=True
        )
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Ensure templates directory exists
    if not os.path.exists("templates"):
        os.makedirs("templates")
        print("Created templates directory. Please put index.html there.")
    
    print("Starting Army of Safeguards Demo Server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)

