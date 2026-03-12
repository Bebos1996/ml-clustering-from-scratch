## ml-clustering-from-scratch
From scratch implementation of the **K-Means clustering algorithm from scratch in Python** for educational purposes as part of a university machine learning course.

---

## Features
This implementation includes:

- K-Means clustering algorithm
- Random initialization of centroids
- Multiple runs to select the best clustering solution
- Elbow method for optimal number of clusters
- Silhouette coefficient computation
- External validation metrics:
  - Purity
  - Rand Index
  - Precision
  - Recall
  - F1 score
- Visualization of cluster assignments
- Silhouette plots

---

## Project Structure
data/ dataset used
src/ source code
results/ generated plots
requirements.txt

---

## Dataset
This project uses the **Iris dataset**, a classic dataset used for clustering and classification tasks.

Features:
- Sepal length
- Sepal width
- Petal length
- Petal width

Classes:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

## Validation Metrics
The project implements several clustering evaluation metrics.
Internal validation
- Silhouette coefficient

External validation
- Purity
- Rand Index
- Precision
- Recall
- F1 score

These metrics allow evaluating how well the clustering corresponds to the true classes.

---

## Visualizations
The following visualizations are generated:
- Elbow method plot
- Silhouette plots
- Scatter plots of feature pairs showing cluster assignments

Example outputs are saved in the results/ folder.

---

## Installation
Create a virtual environment and install the required dependencies:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run the project:
python src/evaluation.py

The script will:
- Run the K-Means algorithm with different values of K
- Prompt the user to choose the best value of K based on the elbow plot
- Compute clustering metrics
- Generate visualizations
- Save plots inside the results/ directory

The manual selection of K is intentional and allows observing how clustering performance changes when the number of clusters is not optimal.

---

## Elbow Method
The elbow method is used to estimate the optimal number of clusters.

For educational purposes, the script displays the elbow plot and asks the user to manually select the best value of **K**.  
This allows exploring how clustering performance metrics (such as purity, Rand index and silhouette score) change when the number of clusters is incorrectly chosen.

---

## Notes
This repository contains a cleaned and reorganized version of an original university machine learning project.