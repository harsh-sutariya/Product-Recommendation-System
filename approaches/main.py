import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

combined_dataset = pd.read_csv('./data/combined_dataset.csv')

def load_saved_models():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)

    with open('cluster_indices.pkl', 'rb') as f:
        cluster_indices = pickle.load(f)

    with open('cluster_centroids.pkl', 'rb') as f:
        cluster_centroids = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('lsa_model.pkl', 'rb') as f:
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

# Example usage
product_title = "metal saw"
num_recommendations = 5
recommended_indices = recommend_products(product_title, num_recommendations)
recommended_products = []

for cluster_id, index, _ in recommended_indices:
    recommended_products.append(combined_dataset.iloc[index])
print("Recommended Products:")
for product in recommended_products:
    print(product['product_title'])
