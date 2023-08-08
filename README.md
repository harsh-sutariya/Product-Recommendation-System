# Product Recommender App

This is a Flask-based web application that provides product recommendations based on product descriptions using clustering and cosine similarity.
Sure, I'd be happy to explain the techniques used in your product recommendation project: K-Means and Latent Semantic Analysis (LSA).

## K-Means Clustering:

**K-Means** is a popular unsupervised machine learning algorithm used for clustering data points into groups or clusters based on their similarity. In your project, K-Means is used to group product descriptions into clusters based on their content. Here's how it works:

1. **Initial Cluster Centers:** K-Means starts by randomly selecting 'k' initial cluster centers (where 'k' is the number of clusters you want to create).

2. **Assign Points to Clusters:** Each data point (in your case, a product description) is assigned to the nearest cluster center based on a distance metric (typically Euclidean distance).

3. **Update Cluster Centers:** The cluster centers are recalculated as the mean of the data points assigned to each cluster.

4. **Repeat:** Steps 2 and 3 are repeated iteratively until the cluster assignments and centers stabilize or a maximum number of iterations is reached.

5. **Final Result:** The final clusters are formed, and each product description belongs to one of these clusters.

In this app, K-Means helps group similar product descriptions together, making it easier to recommend products based on similarity within clusters.

## Latent Semantic Analysis (LSA):

**Latent Semantic Analysis (LSA)** is a technique that falls under the category of dimensionality reduction and text analysis. It's often used to find hidden patterns in text data. Here's how LSA works:

1. **Term-Document Matrix:** LSA starts by creating a term-document matrix where rows represent terms (words) and columns represent documents (product descriptions). Each cell contains a numerical value indicating the importance of that term in the corresponding document.

2. **Singular Value Decomposition (SVD):** LSA applies Singular Value Decomposition to the term-document matrix. SVD decomposes the matrix into three matrices: U, Σ, and V. These matrices capture latent semantic relationships between terms and documents.

3. **Reducing Dimensionality:** The Σ matrix contains singular values, which represent the importance of latent semantic concepts. By truncating or selecting the top 'k' singular values, you can reduce the dimensionality of the term-document matrix.

In the project, LSA is applied to the TF-IDF vectors of the product descriptions. This helps in reducing the dimensionality of the data while preserving the latent semantic information. The reduced-dimensional vectors can be used in clustering (as in your K-Means step) and for calculating similarity between the user query and product descriptions.

Both K-Means and LSA work together to organize the product descriptions into clusters and create a lower-dimensional representation for efficient recommendation based on user queries.

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Features](#features)


## Description

The Product Recommender App uses a combination of clustering and cosine similarity to recommend products to users based on a query. It includes:

- Preprocessing of product descriptions and additional dataset
- TF-IDF vectorization and Latent Semantic Analysis (LSA) for dimensionality reduction
- K-Means clustering for grouping products
- Cosine similarity calculation for recommendation

## Getting Started

### Prerequisites

- Python (>=3.6)
- Flask (install using `pip install Flask`)
- pandas (install using `pip install pandas`)
- scikit-learn (install using `pip install scikit-learn`)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/harsh-sutariya/assignment2.git
   cd product-recommender-app

2. Install the required packages:
	```bash
	pip install -r requirements.txt

## Usage

1. Place your model files (kmeans_model.pkl, cluster_indices.pkl, etc.) in the models directory.

2. Make sure your combined dataset (combined_dataset.csv) is available.

3. Run the Flask app:
	```bash
	python app.py

4. Open a web browser and navigate to http://127.0.0.1:5000/ to use the app.

## Features

Search for products using combined queries (query|product_uid) or just queries
Directly retrieve product titles by searching for a perfect product UID match
Get product recommendations based on query and similarity to clusters
