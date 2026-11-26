import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from hdbscan import validity
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import argparse
import logging
from itertools import product  


def setup_logging():
    """
    Configures logging for the program.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )



def process_param(param_row, X_scaled, id_molecoles, output_dir,fp_source_dir=None,label_output_dir="label_output"):
    """
    Process a single set of parameters:
    - Applies UMAP
    - Runs HDBSCAN
    - Calculates evaluation metrics
    - Saves the resulting embedding
    - Returns the results of the metrics

    Args:
        param_row (dict): Dictionary containing the parameters.
        X_scaled (numpy.ndarray): Standardized data.
        id_molecoles (pandas.Series): ID of the molecules.
        output_dir (str): Directory for saving embeddings.

    Returns:
        dict: Results of metrics and path of embedding.
*** Translated with www.DeepL.com/Translator (free version) ***

.
    """
    try:
        # Extraction and conversion of parameters by column names
        n_neighbors = int(param_row['n_neighbors'])
        min_dist = float(param_row['min_dist'])
        n_components = int(param_row['n_components'])
        min_samples = int(param_row['min_samples'])
        min_cluster_size = int(param_row['min_cluster_size'])
        flag = 0

        # Configuration of UMAP
        try:
            # Loading embedding from a file if specified
            if fp_source_dir is not None:
                fp_file = os.path.join(fp_source_dir, f'clusterable_embedding_n{n_neighbors}_d{min_dist}_c{n_components}.csv')
                logging.info(f"Loading embedding from {fp_file}...")

            if fp_source_dir is not None and os.path.exists(fp_source_dir) and fp_file is not None and os.path.exists(fp_file):
                try:
                    logging.info(f"Reading the fingerprint embedding file: {fp_file}")
                    clusterable_embedding_df = pd.read_csv(fp_file)
                    clusterable_embedding = clusterable_embedding_df.drop('id_molecola', axis=1).values
                    flag = 1
                    logging.info(f"Loading embedding from {fp_file} completed.")
                except Exception as e:
                    logging.info(f"Error while loading fingerprint file: {e}")
                    return {
                        'params': ",".join(map(str, [
                            param_row.get('n_neighbors', 'NA'),
                            param_row.get('min_dist', 'NA'),
                            param_row.get('n_components', 'NA'),
                            param_row.get('min_samples', 'NA'),
                            param_row.get('min_cluster_size', 'NA')
                        ])),
                        'n_clusters': np.nan,
                        'percent_clustered': np.nan,
                        'silhouette': np.nan,
                        'dbcv': np.nan,
                        'embedding_file': None,
                        'error': str(e)
                    }

            else:
                logging.info(f"Embedding file not found: application of UMAP with parameters {param_row}...")
                # Configuring UMAP if not loaded from a file.
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=n_components,
                    random_state=42,
                    metric='cosine'
                )

                # Application of UMAP
                clusterable_embedding = reducer.fit_transform(X_scaled).astype(np.float64)
        except Exception as e:
            logging.info(f"Error when applying UMAP for parameters {param_row}: {e}")
            return {
                'params': ",".join(map(str, [
                    param_row.get('n_neighbors', 'NA'),
                    param_row.get('min_dist', 'NA'),
                    param_row.get('n_components', 'NA'),
                    param_row.get('min_samples', 'NA'),
                    param_row.get('min_cluster_size', 'NA')
                ])),
                'n_clusters': np.nan,
                'percent_clustered': np.nan,
                'silhouette': np.nan,
                'dbcv': np.nan,
                'embedding_file': None,
                'error': str(e)
            }

        # Creation of the DataFrame with embedding and molecule IDs.
        embedding_columns = [f"UMAP_{i+1}" for i in range(clusterable_embedding.shape[1])]
        clusterable_embedding_df = pd.DataFrame(clusterable_embedding, columns=embedding_columns)
        clusterable_embedding_df['id_molecola'] = id_molecoles

        embedding_filename = os.path.join(output_dir, f'clusterable_embedding_n{n_neighbors}_d{min_dist}_c{n_components}.csv')
        # Saving the embedding to a CSV with a distinctive name but only if it was not loaded from a file.
        if flag == 0:
            clusterable_embedding_df.to_csv(embedding_filename, index=False)
            logging.info(f"Saved embedding in '{embedding_filename}'.")

        # Application of HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            metric='euclidean',  # You can change the metrics if necessary
            prediction_data=False
        )
        labels = clusterer.fit_predict(clusterable_embedding)

        # Saving Labels to a CSV
        labels_df = pd.DataFrame(labels, columns=['cluster'])
        labels_df.to_csv(os.path.join(label_output_dir, f'labels_n{n_neighbors}_d{min_dist}_c{n_components}_ms{min_samples}_mcs{min_cluster_size}.csv'), index=False)
        
        # Calculation of the number of clusters (excluding noise)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Calculation of the percentage of clustered points
        clustered = (labels >= 0)
        tot = clustered.sum() / len(labels) * 100

        # Calculation of silhouette score
        if num_clusters > 1:
            silhouette = silhouette_score(clusterable_embedding[clustered], labels[clustered])
        else:
            silhouette = np.nan

        # Calculation of the DBCV index
        if num_clusters > 1:
            try:
                dbcv = validity.validity_index(clusterable_embedding[clustered], labels[clustered])
            except AttributeError as ae:
                logging.error(f"Error when calculating DBCV for parameters {param_row}: {ae}")
                # Print all attributes of the clusterer object to diagnose
                logging.debug(f"Attributes of HDBSCAN: {dir(clusterer)}")
                dbcv = np.nan
        else:
            logging.warning(f"Number of clusters <= 1 for {param_row} parameters, impossible to calculate DBCV.")
            dbcv = np.nan

        # Prepare the results
        result = {
            'params': ",".join(map(str, [
                n_neighbors, min_dist, n_components, min_samples, min_cluster_size
            ])),
            'n_clusters': num_clusters,
            'percent_clustered': tot,
            'silhouette': silhouette,
            'dbcv': dbcv,
            'embedding_file': embedding_filename
        }

        logging.info(f"Completed: {result['params']}")
        return result
    except Exception as e:
        logging.error(f"Error with parameters {param_row}: {e}")
        return {
            'params': ",".join(map(str, [
                param_row.get('n_neighbors', 'NA'),
                param_row.get('min_dist', 'NA'),
                param_row.get('n_components', 'NA'),
                param_row.get('min_samples', 'NA'),
                param_row.get('min_cluster_size', 'NA')
            ])),
            'n_clusters': np.nan,
            'percent_clustered': np.nan,
            'silhouette': np.nan,
            'dbcv': np.nan,
            'embedding_file': None,
            'error': str(e)
        }


def get_max_workers():
    """
    Automatically determines the maximum number of parallel processes based on the physical cores.

    Returns:
        int: Number of parallel processes.
    """
    physical_cores = psutil.cpu_count(logical=False)
    if physical_cores:
        max_workers = physical_cores
    else:
        max_workers = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
    return max_workers

def main(fingerprints_file='../results/molecule_fingerprints2.csv', output_dir='../results/umap_hdbscan_validation_embeddings', results_file='umap_hdbscan_validation_results.csv',fp_source_dir=None,label_output_dir="../results/umap_hdbscan_validation_label_output"):
    """
    Main function that coordinates the clustering process.

    Args:
        params_file (str): Path to the CSV file containing the results.
        filtered_params_file (str): Path to the CSV file of valid parameters.
        fingerprints_file (str): Path to the CSV file containing the fingerprints of the molecules.
        output_dir (str): Directory to save the embeds.
        results_file (str): Name of the CSV file to save the results.
    """


    setup_logging()
    logging.info("Start of the clustering process.")


    # Loading molecule data
    if not os.path.exists(fingerprints_file):
        logging.error(f"The fingerprints file '{fingerprints_file}' does not exist.")
        return

    print(f"Reading fingerprints from: {fingerprints_file}")
    data = pd.read_csv(fingerprints_file)
    
    if 'id_molecola' not in data.columns:
        logging.error("The column 'id_molecule' is not present in the fingerprint file.")
        return

    id_molecoles = data['id_molecola']
    X = data.drop('id_molecola', axis=1)
    logging.info(f"Uploaded data: {X.shape[0]} molecules with {X.shape[1]} fingerprint.")

    # Data standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Standardization of data completed.")

    # Preparing the directory to save embeddings.
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output preparation directory:{output_dir}")

    os.makedirs(label_output_dir, exist_ok=True)
    logging.info(f"Output directory for labels prepared: '{label_output_dir}'.")

    # Determining the maximum number of workers
    max_workers = get_max_workers()
    logging.info(f"Number of parallel processes set to: {max_workers}.")

    # Define parameter ranges
    n_neighbors_range = [10, 15, 20, 25, 30, 35, 40, 45]
    min_dist_range = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
    n_components_range = [20, 25, 30, 35, 40, 45, 50]
    min_samples_range = [5, 10, 20]
    min_cluster_size_range = [30, 50, 70, 100, 150]

    # Generates all parameter combinations as dictionaries.
    param_combinations = [
        {
            'n_neighbors': n,
            'min_dist': d,
            'n_components': c,
            'min_samples': s,
            'min_cluster_size': m
        }
        for n, d, c, s, m in product(
            n_neighbors_range,
            min_dist_range,
            n_components_range,
            min_samples_range,
            min_cluster_size_range
        )
    ]

    #remove if you want to run the full version
    # param_combinations = param_combinations[:2]

    # Preparation for the collection of results
    results = []

    # Initializing the progress bar
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Creating futures by passing each row as a dictionary
        futures = {
            executor.submit(process_param, params, X_scaled, id_molecoles, output_dir,fp_source_dir,label_output_dir): params
            for params in param_combinations
        }

        # Iteration through futures with a progress bar.
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing parameters'):
            result = future.result()
            results.append(result)

    # Creating a DataFrame of the results
    results_df = pd.DataFrame(results)
    logging.info("All parameter sets were processed.")

    # Function to generate a unique file name
    def generate_unique_filename(filename):
        base, extension = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        while os.path.exists(new_filename):
            new_filename = f"{base}_{counter}{extension}"
            counter += 1
        return new_filename

    # Generate a unique filename if the file already exists.
    results_file = generate_unique_filename(results_file)

    # Saving the results to a CSV
    results_df.to_csv(results_file, index=False)
    logging.info(f"Results saved in '{results_file}'.")
    logging.info("Clustering process completed.")

    return results_df

    


def UMAP_HDBSCAN_tuning(fp_file, embeddings_dir, results_file, fp_source_dir=None, label_output_dir="label_output"):
    """
    Function to perform clustering with UMAP and HDBSCAN using pre-calculated or on-the-fly embeddings.

    Args:
        fingerprints_file (str): Path to the CSV file containing molecule fingerprints.
        output_dir (str): Directory to save the embeddings.
        results_file (str): Name of the CSV file to save the results.
        fp_source_dir (str, optional): Directory containing pre-calculated embeddings. Defaults to None.
        label_output_dir (str, optional): Directory to save the labels. Defaults to "label_output".

    Returns:
        pandas.DataFrame: DataFrame containing the clustering results.
    """
    result_df = main(
        fingerprints_file=fp_file, 
        output_dir=embeddings_dir, 
        results_file=results_file, 
        fp_source_dir=fp_source_dir, 
        label_output_dir=label_output_dir
    )
    return result_df

# if __name__ == '__main__':
#     UMAP_HDBSCAN_tuning('new_molecule_fingerprints2.csv')
