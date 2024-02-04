import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Constants for the analysis - can be fine-tuned based on your dataset
N_COMPONENTS = 30  # Number of principal components for PCA dimensionality reduction
METRIC = 'euclidean'  # Metric used for HDBSCAN

# Function to load data and prepare for clustering
def load_and_prepare_data(file_path):
    """
    Load data from CSV, preprocess vector components, and filter for valid vectors.
    :param file_path: Path to the input CSV file.
    :return: Tuple containing the original dataframe and a dataframe of valid vectors.
    """
    df = pd.read_csv(file_path)

    # Convert vector component strings into one cohesive numpy array per row
    df['vector_array'] = df.apply(lambda row: np.array(
        [float(item) for part in range(1, 7)  # Adjust range based on CSV structure
         for item in str(row[str(part)]).split(',') if item.strip() != '']
    ), axis=1)

    # Initialize default cluster assignment
    df['cluster'] = -1

    # Filter rows with vector array lengths that match the expected dimension (e.g., 3072 components)
    valid_vectors_df = df[df['vector_array'].apply(len) == 3072].copy()

    return df, valid_vectors_df

# Function to reduce dimensionality of vector data using PCA
def apply_pca(valid_vectors_df, n_components):
    """
    Apply PCA to reduce the dimensionality of the vector data.
    :param valid_vectors_df: DataFrame containing the high-dimensional vectors.
    :param n_components: Number of principal components to use for PCA.
    :return: The dimensionality reduced data.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(np.stack(valid_vectors_df['vector_array'].values))
    return reduced_data

# Function to perform clustering using HDBSCAN
def perform_clustering_hdbscan(reduced_data):
    """
    Perform clustering on the dimensionality reduced data using HDBSCAN.
    :param reduced_data: The dimensionality reduced data.
    :return: Cluster labels and the HDBSCAN clusterer object.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric=METRIC,
        cluster_selection_epsilon=0.0  # Use the updated epsilon configuration
    )
    labels = clusterer.fit_predict(reduced_data)
    return labels, clusterer

# Function to sort by clusters and save data
def sort_and_save(df, labels, output_path):
    """
    Sort the original dataframe by assigned cluster IDs and save the results.
    :param df: The original dataframe.
    :param labels: The cluster labels assigned to the data points.
    :param output_path: The file path to save the Excel sheet with clustering results.
    """
    df['cluster'] = labels
    df.sort_values(by='cluster', inplace=True)
    output_df = df[['Name', 'cluster']]
    output_df.to_excel(output_path, index=False)

# Main execution: Load data, apply PCA, perform clustering, and save results
if __name__ == "__main__":
    input_csv_path = 'path/to/your/input.csv'  # Input file path
    output_xlsx_path = 'path/to/your/output.xlsx'  # Output file path

    df, valid_vectors_df = load_and_prepare_data(input_csv_path)
    reduced_data = apply_pca(valid_vectors_df, N_COMPONENTS)
    labels, clusterer = perform_clustering_hdbscan(reduced_data)
    sort_and_save(df, labels, output_xlsx_path)

    # Print a summary of analysis parameters and results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Analysis Parameters and Results:")
    print(f"PCA Components: {N_COMPONENTS}")
    print(f"Metric: {METRIC}, Min Cluster Size: 2, Min Samples: 1, Epsilon: 0.0")
    print(f"Number of Clusters: {n_clusters}")

    try:
        silhouette_avg = silhouette_score(reduced_data, labels)
        print(f"Silhouette Score: {silhouette_avg}")
    except Exception as e:
        print(f"Silhouette Score calculation error: {e}")
