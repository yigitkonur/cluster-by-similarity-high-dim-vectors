import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Constants
VECTOR_DIMENSION = 3072  # Define the dimensionality of your vectors
SIMILARITY_THRESHOLD = 0.78  # Set your desired similarity threshold for DBSCAN
MIN_SAMPLES = 2  # Minimum number of samples required for points to form a dense region

def load_and_prepare_data(file_path):
    """
    Load the data from a CSV file and prepare it for clustering.
    This includes creating a numpy array from concatenated vector parts and filtering the data.
    :param file_path: File path to the CSV containing vector data.
    :return: Tuple (original dataframe, dataframe with valid vectors for clustering)
    """
    df = pd.read_csv(file_path)

    # Concatenate vector parts into a single numpy array for each row
    df['vector_array'] = df.apply(
        lambda row: np.array(
            [float(item)
             for part in range(1, 7)  # Adjust range based on the number of vector parts in the CSV
             for item in str(row[str(part)]).split(',')
             if item.strip() != '']
        ),
        axis=1
    )

    # Initialize a default cluster ID for all items
    df['cluster'] = -1

    # Keep only rows with valid vector lengths matching the expected dimensionality
    valid_vectors_df = df[df['vector_array'].apply(len) == VECTOR_DIMENSION].copy()

    return df, valid_vectors_df

def perform_clustering_dbscan(valid_vectors_df, similarity_threshold, min_samples):
    """
    Perform DBSCAN clustering using the crafted similarity threshold and minimum samples.
    :param valid_vectors_df: DataFrame containing valid vectors for clustering.
    :param similarity_threshold: Similarity threshold for the DBSCAN algorithm.
    :param min_samples: Minimum number of samples required by DBSCAN to form a cluster.
    :return: DataFrame with clusters assigned to each valid vector.
    """
    eps_value = 1 - similarity_threshold

    # Apply the DBSCAN clustering algorithm
    db = DBSCAN(eps=eps_value, min_samples=min_samples, metric='cosine')
    valid_vectors_df['cluster'] = db.fit_predict(np.stack(valid_vectors_df['vector_array'].values))

    # Gather statistical data
    labels = valid_vectors_df['cluster']
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_clustered = len(valid_vectors_df) - n_noise
    total_points = len(valid_vectors_df)

    # Calculate noise and clustering ratios, and average cluster size excluding noise
    noise_ratio = n_noise / total_points if total_points else 0
    clustered_ratio = n_clustered / total_points if total_points else 0
    average_cluster_size = n_clustered / n_clusters if n_clusters else 0

    # Remove noise from cluster sizes statistics
    cluster_sizes = valid_vectors_df['cluster'].value_counts().sort_index()
    if -1 in cluster_sizes:
        cluster_sizes = cluster_sizes.drop(-1)

    # Output a summary of the clustering
    print(f"Similarity Threshold: {similarity_threshold}")
    print(f"Min Samples: {min_samples}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Noise Ratio: {noise_ratio:.2%}")
    print(f"Clustered Ratio: {clustered_ratio:.2%}")
    print(f"Average Cluster Size (excluding noise): {average_cluster_size:.2f}")
    print(f"Cluster Sizes: {cluster_sizes.to_dict()}")

    # Calculate silhouette score when possible
    silhouette_avg = 'N/A'  # Default when not computable
    if 1 < n_clusters < len(valid_vectors_df):
        try:
            silhouette_avg = silhouette_score(
                np.stack(valid_vectors_df['vector_array'].values),
                labels,
                metric='cosine'
            )
            print(f"Silhouette Coefficient: {silhouette_avg:.2f}")
        except Exception as e:
            print(f"Silhouette Coefficient calculation failed: {e}")

    return valid_vectors_df

def sort_and_save(df, valid_vectors_df, output_path):
    """
    Sort the DataFrame based on the assigned clusters and save the results into an XLSX file.
    :param df: Original DataFrame.
    :param valid_vectors_df: DataFrame containing the vectors after clustering.
    :param output_path: File path to save the XLSX output.
    """
    # Ensure that the 'cluster' column in the original DataFrame is up to date
    df.loc[valid_vectors_df.index, 'cluster'] = valid_vectors_df['cluster']

    # Sort the DataFrame by the 'cluster' column
    df.sort_values(by='cluster', inplace=True)

    # Select relevant columns for saving (here 'Name' and 'cluster' are provided as examples)
    output_df = df[['Name', 'cluster']]

    # Save to an XLSX file
    output_df.to_excel(output_path, index=False)

# Usage Example
if __name__ == "__main__":
    # Define file paths
    input_csv_path = 'path/to/your/input.csv'   # Replace with the path to your CSV file
    output_xlsx_path = 'path/to/your/output.xlsx'

    # Load data, perform clustering, and save the sorted results
    df, valid_vectors_df = load_and_prepare_data(input_csv_path)
    clustered_valid_vectors_df = perform_clustering_dbscan(valid_vectors_df, SIMILARITY_THRESHOLD, MIN_SAMPLES)
    sort_and_save(df, clustered_valid_vectors_df, output_xlsx_path)
