
# k-NN Wine Dataset Classifier

This project implements a **k-Nearest Neighbors (k-NN)** classifier from scratch using Python, applied to the well-known **Wine dataset**. The model is evaluated using different values of **k** and three distance metrics: **Euclidean**, **Manhattan**, and **Chebyshev** (optional).

- `knn.py`: Custom implementation of the k-NN algorithm with support for different distance metrics.
- `analysis.ipynb`: Jupyter Notebook containing data analysis, training results, accuracy plots, and confusion matrices. Desired graphs can be drawn by uncommenting the relevant code cells.
- `README.md`: This file — contains project description and instructions.
- `wine.data`, `wine.names`: UCI Wine Dataset files.
- `figures/`: Contains all visual outputs such as accuracy vs k plots and confusion matrices.

The Wine dataset consists of **178 samples**, each with **13 chemical attributes** derived from the analysis of wines grown in the same region in Italy but derived from three different cultivars (classes).

- **Features**: Alcohol, Malic acid, Ash, Magnesium, Flavanoids, Hue, etc.
- **Target Classes**: 1, 2, 3

The dataset used in this project is sourced from **scikit-learn** and matches the UCI Wine dataset.

Setup Instructions

To run the code in **Google Colab**, follow these steps:

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
