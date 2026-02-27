# Executive Team Quality (Qual) Index Construction via PCA

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-lightgrey)

## 📌 Project Overview
This repository provides a rigorously formatted Principal Component Analysis (PCA) pipeline designed for academic econometrics and management research. It extracts unobservable latent constructs (e.g., Executive Team Quality, *Qual*) from a multidimensional set of standardized demographic and background variables. 

The output includes publication-ready, grayscale visual diagnostics: a Scree Plot with an 85% cumulative variance heuristic threshold, and a geometric Vector Loading Plot.



## 🛠 Environmental Requirements
- Python 3.8+
- Scikit-learn
- Matplotlib
- Pandas
- Numpy
- Openpyxl (For robust `.xlsx` ingestion)

## 🚀 Installation

```bash
git clone [https://github.com/your-username/academic-pca-index.git](https://github.com/your-username/academic-pca-index.git)
cd academic-pca-index
pip install -r requirements.txt
```

## 📊 Usage Guide

1. Ensure your pre-standardized data (`data.xlsx`) is placed in the root directory. The script defaults to scanning `Sheet2` for variables prefixed with `X1` to `X5`.
2. Run the construction script:
   ```bash
   python src/pca_index_construction.py
   ```
3. The script will output:
   - A high-fidelity vector graphic: `Fig_PCA_Index_Construction.svg`
   - The exact linear combination formula for manuscript inclusion printed in the terminal.

## 🔬 Methodology

To mitigate multicollinearity among executive characteristics and construct a parsimonious index, we employ Principal Component Analysis (PCA). Let $X_1, X_2, \dots, X_p$ represent the standardized input features. The first principal component ($PC_1$), which captures the maximum variance, is defined as a linear combination of the underlying variables:

$$ PC_1 = \sum_{j=1}^{p} a_{1j} X_j $$

where the loading vector $a_1 = (a_{11}, a_{12}, \dots, a_{1p})^T$ is the eigenvector corresponding to the largest eigenvalue $\lambda_1$ of the sample covariance matrix, subject to the constraint $\sum a_{1j}^2 = 1$. The resulting continuous scalar, $PC_1$, serves as our proxy variable for the overall unobserved Executive Team Quality (*Qual*).
Plaintext
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
openpyxl>=3.0.0
Plaintext
# Environments
.env
.venv
env/
venv/
ENV/

# Python Specific
__pycache__/
*.py[cod]
*$py.class

# Data sets (Strict Anonymity)
*.csv
*.xlsx
*.xls
data/
raw_data/
processed_data/

# Generated Outputs
*.svg
*.png
*.pdf

# IDEs
.idea/
.vscode/
*.DS_Store
