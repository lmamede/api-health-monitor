# API Health Monitor

This repository contains the full implementation, exploratory analysis, experimental pipeline, and result generation scripts used in the paper Statistical Trust Modeling for Endpoint-level Network Anomaly Detection. The project focuses on anomaly detection and robustness evaluation in API traffic monitoring under non-stationary conditions.

## Repository Structure
```
api-health-monitor/
│
├── notebooks/
│ ├── 1-eda/ # Exploratory Data Analysis
│ │ ├── outputs/ # Generated plots and intermediate artifacts
│ │ ├── scripts/ # Data preprocessing and EDA utilities
│ │ ├── EDA-APA_DDoS.ipynb
│ │ ├── EDA-CICDDoS_2019.ipynb
│ │ └── EDA-ToN_IoT.ipynb
│ │
│ ├── 2-experiments/ # Experimental execution pipeline
│ │ ├── outputs/ # Experimental outputs
│ │ ├── scripts/ # Experiment logic and evaluation functions
│ │ └── Experiment.ipynb
│ │
│ └── 3-results/ # Post-processing and result analysis
│ ├── outputs/ # Final metrics and figures
│ ├── scripts/ # Result aggregation utilities
│ └── Result_Analysis.ipynb
│
├── requirements.txt # Python dependencies
└── .gitignore
```

## Environment Setup

We recommend using Python 3.14.+.

### Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```` 

#### Install dependencies
```bash
pip install -r requirements.txt
```

#### Reproducing the Results
To reproduce all results reported in the paper, follow the pipeline below.

**Step 1 — Dataset Preparation**

Place the datasets inside: notebooks/0-datasets/

Then run the EDA notebooks: notebooks/1-eda/EDA-*.ipynb

This step performs:
- Data cleaning
- Feature preparation
- Exploratory visualization
- Intermediate artifact generation

**Step 2 — Run Experiments**

Execute: notebooks/2-experiments/Experiment.ipynb

This notebook:
- Trains the models
- Applies trust estimation
- Computes degradation metrics (e.g., Ra)
- Stores outputs under 2-experiments/outputs/

**Step 3 — Generate Final Results**

Execute: notebooks/3-results/Result_Analysis.ipynb

This step:
- Aggregates metrics
- Produces final plots
- Generates evaluation tables used in the paper
- Outputs are stored under: notebooks/3-results/outputs/

#### Experimental Design

The experiments evaluate:
- Robustness under non-stationary traffic
- Performance degradation metrics
- Endpoint-level behavior
- Parametric sensitivity analysis

All reported figures in the paper can be regenerated from the provided notebooks.

#### Reproducibility Notes

All experiments were executed with fixed random seeds. Intermediate outputs are stored to ensure traceability.
