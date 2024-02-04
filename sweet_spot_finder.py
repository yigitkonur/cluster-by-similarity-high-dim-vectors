import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# Function to load data and prepare vectors for clustering
def load_and_prepare_data(file_path):
    # Load data from a CSV file
    df = pd.read_csv(file_path)
    
    # Preprocess the data: Generate a single numpy array per row by concatenating vector parts // change it based on your needs but in my use case I have to seperate them into 6 parts to work in Excel.
    df['vector_array'] = df.apply(lambda row: np.array([float(item) 
                                                         for part in range(1, 7)  # Assuming 6 parts of vectors
                                                         for item in str(row[str(part)]).split(',') 
                                                         if item.strip() != '']),
                                  axis=1)

    # Initialize all rows with a default cluster ID of -1 (indicating no cluster)
    df['cluster'] = -1

    # Filter rows to include only those with a vector array size that matches the model's output dimensions
    vector_dimension = 3072  # Example dimension for text-ada-003-large embedddings
    valid_vectors_df = df[df['vector_array'].apply(len) == vector_dimension].copy()

    return df, valid_vectors_df

# Function to perform DBSCAN clustering on preprocessed data
def perform_clustering_dbscan(valid_vectors_df, similarity_threshold, min_samples):
    # Convert similarity threshold to DBSCAN's eps (epsilon) parameter 
    eps_value = 1 - similarity_threshold

    # Initialize DBSCAN with specified parameters
    db = DBSCAN(eps=eps_value, min_samples=min_samples, metric='cosine')
    
    # Fit the DBSCAN model and assign cluster labels
    valid_vectors_df['cluster'] = db.fit_predict(np.stack(valid_vectors_df['vector_array'].values))

    # Extract the unique cluster labels
    labels = valid_vectors_df['cluster']
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Output the results of clustering
    print(f"Similarity Threshold: {similarity_threshold:.3f}, Min Samples: {min_samples} -> Number of clusters: {n_clusters}")

    return valid_vectors_df

# Main execution: Adjust parameters and file path as necessary
if __name__ == "__main__":
    input_csv_path = 'path/to/your/input.csv'  # Replace with your actual file path
    min_samples = 2  # Define the minimum number of samples in a neighborhood for point classification

    # Iterate over a range of similarity thresholds to find the optimal clustering configuration
    for threshold in np.arange(0.995, 0.800, -0.005):  # Adjust the range as needed
        df, valid_vectors_df = load_and_prepare_data(input_csv_path)
        perform_clustering_dbscan(valid_vectors_df, threshold, min_samples)
