# Shopper Spectrum

**Customer Segmentation and Product Recommendations in E-Commerce**

An end-to-end unsupervised learning project that segments e-commerce customers using RFM (Recency, Frequency, Monetary) analysis and recommends products using item-based collaborative filtering — deployed as an interactive Streamlit app.

**Live app:** [shopper-spectrum-app.streamlit.app](https://shopper-spectrum-app.streamlit.app)

---

## Overview

This project analyzes one year of online retail transaction data (541,909 rows) to:

1. **Segment customers** into four behavioral groups — High-Value, Regular, Occasional, At-Risk — using RFM features and clustering.
2. **Recommend products** for a given item using item-based collaborative filtering (cosine similarity on co-purchase patterns).

Both modules, plus supporting analytics, are served through a Streamlit app with a custom console-style UI (dark/light toggle included).

---

## Data Pipeline

The raw dataset (541,909 rows, 8 columns) was cleaned as follows:

| Step | Action |
|---|---|
| 1 | Dropped rows with missing `CustomerID` (~24.9% of rows — no customer to attribute RFM to) |
| 2 | Dropped rows with missing `Description` (~0.27% of rows) |
| 3 | Normalized `Description` text (stripped whitespace, upper-cased) for consistent matching downstream |
| 4 | Removed cancelled invoices (`InvoiceNo` starting with `C`) |
| 5 | Removed rows with non-positive `Quantity` |
| 6 | Removed rows with non-positive `UnitPrice` |
| 7 | Parsed `InvoiceDate` to datetime, cast `CustomerID` to int |
| 8 | Derived `TotalPrice = Quantity × UnitPrice` |
| 9 | Dropped remaining exact duplicate rows |

**Result: 392,692 clean transaction rows across 4,338 unique customers.**

---

## Customer Segmentation

### RFM + Extended Features

Recency, Frequency, and Monetary were computed per customer, with outliers capped using the IQR method and features standardized before clustering. Two extended features (Tenure, Average Order Value) were engineered for deeper post-hoc profiling.

### Model Comparison

Four clustering algorithms were fit and evaluated on the same standardized RFM space:

| Model | Clusters Found | Silhouette Score ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Noise Points |
|---|---|---|---|---|---|
| **KMeans** | 4 | **0.4743** | **0.7951** | **6411.50** | 0 |
| Agglomerative (Ward) | 4 | 0.4402 | 0.7770 | 4916.59 | 0 |
| Gaussian Mixture | 4 | 0.1478 | 1.5307 | 1881.80 | 0 |
| DBSCAN | 2 | 0.2950 | 0.7855 | 16.72 | 16 |

**KMeans (k=4) was selected as the final model** — it produced the highest Silhouette and Calinski-Harabasz scores among the four candidates, and cleanly separated four well-populated, well-balanced clusters (DBSCAN collapsed most points into a single dense cluster instead of four meaningful segments).

Model robustness was additionally checked via bootstrap resampling (10 resamples) and a held-out new-customer prediction test, both confirming stable cluster assignments.

### Final Segments

| Segment | Customers | % of Base | Avg Recency (days) | Avg Frequency | Avg Monetary (£) | % of Total Revenue |
|---|---|---|---|---|---|---|
| **High-Value** | 569 | 13.1% | ~20 | ~9.9 | highest | **38.2%** |
| **Regular** | 853 | 19.7% | ~40 | ~5.0 | mid-high | 33.1% |
| **Occasional** | 1,907 | 44.0% | ~53 | ~2.0 | mid-low | 20.5% |
| **At-Risk** | 1,009 | 23.3% | ~248 | ~1.4 | lowest | 8.2% |

High-Value customers are just over 1 in 8 of the customer base but generate more than a third of total revenue — the core justification for prioritizing retention and targeted offers toward this segment.

---

## Product Recommendation System

Built as **item-based collaborative filtering**: a customer-product purchase matrix is transposed so each product is represented by its purchase pattern across customers, and cosine similarity is computed between products. Given a product name, the top-N most similar products (by co-purchase pattern) are returned. A secondary, purely content-based recommender (TF-IDF over product descriptions) is also built for comparison.

### Offline Evaluation (Leave-One-Out)

One purchased item per eligible customer was held out and the recommender was asked to recover it in its Top-5 recommendations, evaluated against a "most popular items" baseline:

| Approach | Precision@5 | Recall@5 |
|---|---|---|
| **Item-Based Collaborative Filtering** | **0.0104** | **0.0520** |
| Popularity Baseline (Top-5 global) | 0.0024 | 0.0120 |

- Evaluated on 4,248 customers.
- **Coverage: 32.0%** (1,237 of 3,866 products appeared in at least one Top-5 list).
- The collaborative filter outperforms the popularity baseline by roughly 4x on both metrics, confirming it captures genuine co-purchase signal rather than just surfacing bestsellers — though absolute precision is low in a fairly typical way for large, sparse retail catalogues, and coverage indicates a moderate popularity-bias in which recommendations skew toward frequently co-purchased items.

---

## Repository Structure

```
Shopper-Spectrum/
├── app.py                     # Streamlit application (segmentation + recommendation UI)
├── Shopper_Spectrum.ipynb     # Full analysis: EDA, feature engineering, modeling, evaluation
├── online_retail.csv          # Raw transaction dataset (not tracked in git — see .gitignore)
├── requirements.txt           # Python dependencies
├── models/
│   ├── kmeans_model.pkl       # Trained KMeans model (k=4)
│   ├── rfm_scaler.pkl         # StandardScaler fit on RFM features
│   ├── cluster_label_map.pkl  # Maps KMeans cluster index → segment name
│   ├── cosine_sim_df.pkl      # Product-product cosine similarity matrix
│   └── rfm_segments.csv       # Final per-customer RFM + segment table (4,338 customers)
├── images/                    # Saved chart exports from the notebook
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/Mohit-1307/Shopper-Spectrum.git
cd Shopper-Spectrum
pip install -r requirements.txt
streamlit run app.py
```

The app expects the trained artifacts (`kmeans_model.pkl`, `rfm_scaler.pkl`, `cluster_label_map.pkl`, `cosine_sim_df.pkl`, `rfm_segments.csv`) inside `models/`. These are produced by running `Shopper_Spectrum.ipynb` end-to-end, or can be used as already provided in this repo.

---

## Tech Stack

- **Data / ML:** pandas, numpy, scikit-learn (KMeans, Agglomerative, GMM, DBSCAN), scipy
- **Visualization:** matplotlib, seaborn, plotly
- **App:** Streamlit
- **Model persistence:** joblib

---

# Author

**MOHIT SINGH RAJPUT — AI/ML Engineer**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohitsingh1307)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Mohit-1307)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/mohitsinghrajput1307)
[![LeetCode](https://img.shields.io/badge/LeetCode-181717?style=flat-square&logo=leetcode&logoColor=FFA116)](https://leetcode.com/u/MOHIT_SINGH_RAJPUT/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mohitsinghrajput1307@gmail.com)

---

# Acknowledgements

* UCI Online Retail Dataset
* Scikit-Learn
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Open Source Community

---

<div align="center">

*If this project was useful, a ⭐ on the repository is appreciated.*

</div>