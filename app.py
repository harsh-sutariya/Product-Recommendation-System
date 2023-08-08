from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load the combined dataset
combined_dataset = pd.read_csv('./data/combined_dataset.csv')

def load_saved_models():
    model_path = './models/'

    with open(os.path.join(model_path, 'kmeans_model.pkl'), 'rb') as f:
        kmeans = pickle.load(f)

    with open(os.path.join(model_path, 'cluster_indices.pkl'), 'rb') as f:
        cluster_indices = pickle.load(f)

    with open(os.path.join(model_path, 'cluster_centroids.pkl'), 'rb') as f:
        cluster_centroids = pickle.load(f)

    with open(os.path.join(model_path, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(model_path, 'lsa_model.pkl'), 'rb') as f:
        lsa = pickle.load(f)

    return kmeans, cluster_indices, cluster_centroids, vectorizer, lsa


def recommend_products(query_product_title, num_recommendations=5):
    kmeans, cluster_indices, cluster_centroids, vectorizer, lsa = load_saved_models()

    query_vector = vectorizer.transform([query_product_title])
    query_vector_lsa = lsa.transform(query_vector)
    
    predicted_clusters = kmeans.predict(query_vector_lsa)
    
    recommended_products = []
    for cluster_id in predicted_clusters:
        cluster_products = cluster_indices.get(cluster_id, [])
        similarity_scores = cosine_similarity(query_vector_lsa, [cluster_centroids[cluster_id]])[0]
        top_indices = similarity_scores.argsort()[-num_recommendations:][::-1]
        recommended_products.extend([(cluster_id, cluster_products[i], similarity_scores[i]) for i in top_indices])
    
    return recommended_products

@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_products = []

    if request.method == 'POST':
        user_input = request.form['query']
        query_product_title, product_uid = user_input.split('|') if '|' in user_input else (user_input, None)

        # Search for a perfect match in the "B" column of the combined dataset
        perfect_match_product = combined_dataset[combined_dataset['product_uid'] == product_uid]

        if not perfect_match_product.empty:
            recommended_products.append(perfect_match_product.iloc[0]['product_title'])
        else:
            num_recommendations = 5
            recommended_indices = recommend_products(query_product_title, num_recommendations)

            for cluster_id, index, _ in recommended_indices:
                recommended_products.append(combined_dataset.iloc[index]['product_title'])

    return render_template('index.html', recommended_products=recommended_products)

if __name__ == '__main__':
    app.run(debug=True)
