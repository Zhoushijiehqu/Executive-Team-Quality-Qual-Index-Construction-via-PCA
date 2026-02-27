import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import font_manager
import os

# ==========================================
# 1. Academic Visualization Setup
# ==========================================
FONT_PATH = 'TimesSimSunRegular.ttf'

try:
    if os.path.exists(FONT_PATH):
        prop = font_manager.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.sans-serif'] = [prop.get_name()]
        print(f"[UI Setup] Successfully loaded academic font: {FONT_PATH}")
    else:
        print(f"[UI Setup] Font not found. Falling back to system defaults.")
        plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial'] 
except Exception as e:
    print(f"[UI Setup] Font loading exception: {e}")

plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['figure.dpi'] = 300            

# Top-tier journal grayscale palette
COLOR_PRIMARY_BLACK = '#000000'      
COLOR_DARK_GRAY = '#404040'  
COLOR_MEDIUM_GRAY = '#808080'       
COLOR_LIGHT_GRID = '#F0F0F0' 
COLOR_WHITE = '#FFFFFF'      

# ==========================================
# 2. Data Ingestion & Preprocessing
# ==========================================
DATA_PATH = 'data.xlsx' 

try:
    df = pd.read_excel(DATA_PATH, sheet_name='Sheet2')
except FileNotFoundError:
    # Dummy data generation for workflow validation
    print("[Data Pipeline] File not found. Generating dummy dataset.")
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 5), 
                      columns=['X1_Edu', 'X2_Title', 'X3_Tenure', 'X4_Age', 'X5_Entre'])

# Core components (Standardized variable placeholders)
features = ['X1_Edu', 'X2_Title', 'X3_Tenure', 'X4_Age', 'X5_Entre']
feature_labels = ['Education', 'Prof_Title', 'Tenure', 'Age', 'Entrepreneurship']

# Extract numeric matrix and drop NaNs
X = df[features].dropna().values

# ==========================================
# 3. Principal Component Analysis (PCA)
# ==========================================
pca = PCA()
pca.fit(X)

expl_var_ratio = pca.explained_variance_ratio_
cumulative_var_ratio = np.cumsum(expl_var_ratio)
loadings = pca.components_.T 

# ==========================================
# 4. Academic Plot Generation
# ==========================================
fig = plt.figure(figsize=(12, 5.5)) 

font_prop = font_manager.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None
font_kw = {'fontproperties': font_prop} if font_prop else {}

# ------------------------------------------------------------------
# Subplot 1: Scree Plot (Variance Contribution)
# ------------------------------------------------------------------
ax1 = fig.add_subplot(1, 2, 1)
x_ticks = range(1, len(features) + 1)

# Dynamically assign labels based on max absolute loading for interpretability
pc_labels = []
for i in range(len(features)):
    max_idx = np.argmax(np.abs(pca.components_[i]))
    pc_labels.append(f"PC{i+1}\n({feature_labels[max_idx]})")

# Bar plot: Individual explained variance
bars = ax1.bar(x_ticks, expl_var_ratio, 
              color=COLOR_MEDIUM_GRAY, alpha=0.5, edgecolor=COLOR_PRIMARY_BLACK, linewidth=0.8,
              label='Individual Variance', width=0.55)

# Line plot: Cumulative explained variance
ax1.plot(x_ticks, cumulative_var_ratio, 
         color=COLOR_PRIMARY_BLACK, linewidth=1.2, linestyle='-',
         marker='o', markersize=7, markerfacecolor=COLOR_WHITE, 
         markeredgecolor=COLOR_PRIMARY_BLACK, markeredgewidth=1.2,
         label='Cumulative Variance', zorder=10) 

# Value annotations for bars
for rect in bars:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontsize=9.5, color=COLOR_PRIMARY_BLACK, **font_kw)

# 85% Heuristic Threshold Line
ax1.axhline(y=0.85, color=COLOR_DARK_GRAY, linestyle='--', linewidth=1.0, alpha=0.8)
ax1.text(len(features) + 0.1, 0.85, '85% Threshold', ha='left', va='center', 
         fontsize=9, color=COLOR_DARK_GRAY, **font_kw)

# Axes settings
ax1.set_ylabel('Explained Variance Ratio', fontsize=11, **font_kw)
ax1.set_title('(a) Principal Component Selection', y=-0.18, fontsize=12, **font_kw)
ax1.set_ylim(0, 1.15)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(pc_labels, fontsize=9.5, rotation=0, **font_kw)

# Aesthetic cleanup
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', linestyle=':', color=COLOR_LIGHT_GRID)
ax1.legend(loc='upper left', frameon=False, fontsize=9.5, prop=font_prop)

# ------------------------------------------------------------------
# Subplot 2: Loading Plot (Feature Vectors)
# ------------------------------------------------------------------
ax2 = fig.add_subplot(1, 2, 2)

# Central crosshairs
ax2.axhline(0, color=COLOR_DARK_GRAY, linestyle=':', linewidth=0.8)
ax2.axvline(0, color=COLOR_DARK_GRAY, linestyle=':', linewidth=0.8)

# Feature vectors
for i, (x, y) in enumerate(zip(loadings[:, 0], loadings[:, 1])):
    ax2.arrow(0, 0, x, y, 
              color=COLOR_PRIMARY_BLACK, alpha=0.9, 
              head_width=0.04, head_length=0.06, linewidth=1.2,
              length_includes_head=True)
    
    # Quadrant-aware label positioning
    ha = 'left' if x >= 0 else 'right'
    va = 'bottom' if y >= 0 else 'top'
    offset = 1.15
    ax2.text(x * offset, y * offset, feature_labels[i], 
             fontsize=10.5, color=COLOR_PRIMARY_BLACK, ha=ha, va=va, weight='normal', **font_kw)

# Axes settings
ax2.set_xlabel(f'First Principal Component (PC1) - {expl_var_ratio[0]:.1%}', fontsize=10.5, **font_kw)
ax2.set_ylabel(f'Second Principal Component (PC2) - {expl_var_ratio[1]:.1%}', fontsize=10.5, **font_kw)
ax2.set_title('(b) Variable Loading Distribution', y=-0.18, fontsize=12, **font_kw)

# Aspect ratio lock (circular representation constraint)
limit = np.max(np.abs(loadings[:, :2])) * 1.4
ax2.set_xlim(-limit, limit)
ax2.set_ylim(-limit, limit)
ax2.set_aspect('equal')

for spine in ax2.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color(COLOR_PRIMARY_BLACK)

# ==========================================
# 5. Export & Text Outputs
# ==========================================
plt.subplots_adjust(wspace=0.3, bottom=0.2) 
output_file = 'Fig_PCA_Index_Construction.svg'
plt.savefig(output_file, format='svg', bbox_inches='tight')
print(f"\n[Output] Academic figures generated successfully: {output_file}")

# Generate mathematical formula for manuscript inclusion
print("\n[Methodology] Exec. Team Quality (Qual) Construction Formula:")
formula_parts = []
for label, weight in zip(feature_labels, pca.components_[0]):
    sign = "+" if weight >= 0 else "-"
    formula_parts.append(f"{sign} {abs(weight):.3f} \\times \\text{{{label}}}")
print(f"Qual_{{PC1}} = " + " ".join(formula_parts))
