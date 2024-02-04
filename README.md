Clustering high-dimensional data such as the embeddings from language models poses unique challenges. Unlike K-Means, where you need to specify the number of clusters beforehand, this repository offers an advanced solution using the DBSCAN algorithm—a more adaptable and insightful method, especially useful for large and complex datasets.

### What Does the Script Do? 
This collection of scripts:

1. Loads the CSV file containing high-dimensional vectors.
2. Utilizes the DBSCAN algorithm to cluster entities based on their similarity without the need for predefining a specific number of clusters.
3. Identifies and separates outliers or noise from the main clusters to prevent the merging of dissimilar entities.
4. Exports the results with assigned cluster IDs for each entity to an Excel file for convenient review and analysis.

### Main Features
- **Automated Cluster Detection**: Finds the natural number of clusters in the data without any preset conditions.
- **Customization**: Adjustable similarity thresholds and minimum sample sizes to fine-tune the clustering process.
- **High-Dimensional Data Handling**: Developed to work with embeddings from models such as OpenAI's text-ada-003-large.
- **Noise and Outlier Management**: Isolates less similar vectors effectively, maintaining cleaner and more meaningful clusters.
- **Silhouette Score Assessment**: Provides an option to measure the clustering quality using the silhouette score, where feasible.

### How to Use the Main Script
1. Install the necessary Python packages pandas, numpy, scikit-learn, and openpyxl.
2. Set your CSV file path to `input_csv_path`, where your embeddings are stored.
3. Run the script. It will automatically perform clustering, identify the noise, and save the results.

![CleanShot 2024-02-04 at 16 32 44@2x](https://github.com/yigitkonur/high-dimension-dbscan-embedding-clusterer-for-text-ada-003/assets/9989650/61817540-8dfc-48c8-a171-738d72816ea6)  

### sweet_spot_finder.py
The `sweet_spot_finder.py` script assists in finding the optimal DBSCAN parameters by testing different combinations of similarity thresholds and minimum samples. It runs multiple iterations of the clustering process in parallel and reports the number of clusters formed for each configuration. This helps in identifying the "sweet spot," where the clustering logic best aligns with the natural structure of the data.

#### How to Use sweet_spot_finder.py
1. Set the input CSV file path by changing `input_csv_path` in the script.
2. Review and adjust the ranges for `similarity_thresholds` and `min_samples_values` to fit your dataset and clustering goals.
3. Execute the script. The output will display different configurations and their corresponding number of clusters, aiding you in selecting the best parameters for `DBSCAN`.

https://github.com/yigitkonur/high-dimension-dbscan-embedding-clusterer-for-text-ada-003/assets/9989650/f7347b7a-b989-45a4-b2d0-37cec33b7e77

