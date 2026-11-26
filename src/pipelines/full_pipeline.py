import time
"""
This script runs the whole pipeline from the beginning
the executin of this script will require ... hours and will re execute the whole training
and the whole parameters tuning.
"""
import os
import sys

# Path assoluti (non relativi al cwd): il dataset featurizer usa un ProcessPoolExecutor con
# mp_context="spawn", che ricostruisce questo script come modulo "__mp_main__" in ogni processo
# worker. Se questi path fossero relativi ('./src'), risolverebbero in modo sbagliato appena il
# cwd cambia (vedi os.chdir sotto), rompendo gli import in ogni worker.
# Lo script sta in src/pipelines/: la radice del repo e' due livelli sopra. I moduli sono divisi
# in sottocartelle di src/ e si importano per nome, quindi ognuna va in sys.path.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_REPO_DIR, 'src')
for _sub in ('gnn', 'analysis', 'paper', 'explain'):
    sys.path.append(os.path.join(_SRC_DIR, _sub))

def run_pipeline():
    # Import pesanti (tensorflow/jax/optuna/umap/hdbscan via validation_train e
    # apply_clustering_method) tenuti DENTRO la funzione e non a livello di modulo: cosi' i worker
    # "spawn" del ProcessPoolExecutor (che rieseguono il modulo come "__mp_main__", non "__main__")
    # non li reimportano inutilmente ad ogni avvio.
    from post_proc_pipeline import crea_fingerprint, pp_pipeline
    from validation_train import param_fine_tuning
    from train import training
    from predict import predict_drug
    from apply_clustering_method import apply_clustering, centroids_extraction
    from generate_centroids_with_inchikey import generate_centroids_with_inchikey
    from smiles_conversion import smiles_to_inchikey
    from Centroids_to_TMMC import centroids_to_TMMC  # prima mancava: causava un NameError a fine pipeline
    from edgeshaper_step import run_edgeshaper_step

    os.chdir(_SRC_DIR)  # path assoluto: idempotente anche se richiamato piu' volte

    out_dir = '../results'
    data_dir = '../data'
    

    # redefinition of some paths to run just a part of the pipeline REMEMBER TO CLEAN
    #frags_path = f'{out_dir}/test_data/TCMBANK/high_attention_frags.csv'
    #pred_path = f'{out_dir}/test_data/TCMBANK/TMMC_molecule_predictions.csv'

    os.makedirs(f"{out_dir}", exist_ok=True)

    print("Starting GNN Drug-Likeness pipeline")
    print("################################# \n")
    print("################################# \n")


    # File to save execution times
    timing_file = f'{out_dir}/execution_times.txt'
    with open(timing_file, 'w') as f:
        f.write("Execution Times:\n")

    # Perform validation and get the best trial
    
    print("Starting parameter validation")
    print("################################# \n\n\n")
    start_time = time.time()

    best_trial = param_fine_tuning()

    end_time = time.time()
    validation_time = end_time - start_time
    with open(timing_file, 'a') as f:
        f.write(f"Validation Time: {validation_time:.2f} seconds\n")
    params = best_trial.params
    # Save the best parameters found during validation
    with open(f'{out_dir}/validation_train_best_params.txt', 'w') as f:
        f.write(f"{params}")
    print(f"The best parameters are: {params}")

    print("################################# \n\n\n")


    training_epochs = 700 # correct value = 700

    print("Starting training")
    print("################################# \n\n\n")
    # Perform training with the best parameters
    start_time = time.time()

    loss, model_id, experiment_id = training(params, epochs=training_epochs)

    end_time = time.time()
    training_time = end_time - start_time
    with open(timing_file, 'a') as f:
        f.write(f"Training Time: {training_time:.2f} seconds\n")
    print(f"Training completed with loss: {loss}, experiment_id: {experiment_id}, and model_id: {model_id}")
    print("################################# \n\n\n")

    print("Starting model testing")
    print("################################# \n\n\n")
    small = False
    # Perform model testing
    start_time = time.time()

    _, _ = predict_drug('data/zinc/splits/new2_ZINC_final_balanced_test_set.csv', model_id, experiment_id, False, small)

    end_time = time.time()
    testing_time = end_time - start_time
    with open(timing_file, 'a') as f:
        f.write(f"Testing Time: {testing_time:.2f} seconds\n")
    print("Model testing completed")
    print("################################# \n\n\n")


    
    print("Applying the model to final data")
    print("################################# \n\n\n")


    # Perform model testing on final data
    # final flag is set to True to indicate that the model is being applied to the TCM dataset now!
    start_time = time.time()

    ############################################################################################################################################################################
    # small TRUE WHEN TESTING FOR TECHNICAL REASONS ONLY (COULD GIVE SOME ERRORS LATER BECAUSE OF THE SMALL DATASET AND LACK OF POSITIVE SAMPLES)
    ############################################################################################################################################################################
    _, pred_path = predict_drug('data/tm-mc2/preprocessed_TMMC_compounds.csv', model_id, experiment_id, False, small)

    end_time = time.time()
    final_testing_time = end_time - start_time
    with open(timing_file, 'a') as f:
        f.write(f"Final Testing Time: {final_testing_time:.2f} seconds\n")
    print("Model testing on final data completed")
    print("################################# \n\n\n")


    # Fragment attribution with EdgeSHAPer.
    # predict_drug restituisce anche un file di frammenti ricavato dai pesi di attenzione:
    # quello non si usa piu'. I frammenti su cui lavora il resto della pipeline vengono da
    # qui, con lo stesso formato di colonne, quindi nulla a valle cambia.
    print("Starting EdgeSHAPer fragment attribution")
    print("################################# \n\n\n")
    start_time = time.time()

    frags_path = run_edgeshaper_step('data/tm-mc2/preprocessed_TMMC_compounds.csv',
                                     pred_path, out_dir)

    end_time = time.time()
    attribution_time = end_time - start_time
    with open(timing_file, 'a') as f:
        f.write(f"EdgeSHAPer Attribution Time: {attribution_time:.2f} seconds\n")
    print("EdgeSHAPer fragment attribution completed")
    print("################################# \n\n\n")


        # Clustering tuning
    print("Starting clustering tuning")
    print("################################# \n\n\n")
    start_time = time.time()

    reduction_method, cluster_method, best_param, best_silhouette, best_dbcv, frammento_to_index = pp_pipeline(False, frag_file=frags_path, fp_path=f'{out_dir}/new_molecule_fingerprints2.csv',out_dir=out_dir)
    # Save tuning results
    with open(f'{out_dir}/best_valid_clustering_params.txt', 'w') as f:
        f.write(f"Reduction Method: {reduction_method}\n")
        f.write(f"Cluster Method: {cluster_method}\n")
        f.write(f"Best Parameters: {best_param}\n")
        f.write(f"Best Silhouette Score: {best_silhouette}\n")
        f.write(f"Best DBCV Score: {best_dbcv}\n")
        f.write(f"Fragment to Index: {frammento_to_index}\n")
    end_time = time.time()
    clustering_time = end_time - start_time
    with open(timing_file, 'a') as f:
        f.write(f"Clustering Time: {clustering_time:.2f} seconds\n")
    print("Clustering tuning completed")
    print("################################# \n\n\n")

    # Apply the best clustering method
    print("Applying the best clustering method")
    print("################################# \n\n\n")

    if not os.path.exists(f'{out_dir}/final_clustering'):
        os.makedirs(f'{out_dir}/final_clustering')
    start_time = time.time()

    labels_df, clusterer, embeddings_df = apply_clustering(reduction_method, cluster_method, f'{out_dir}/new_molecule_fingerprints2.csv', f'{out_dir}/final_clustering', best_param)


    print(f"Saving embeddings_df results in {out_dir}/final_clustering")
    with open(f'{out_dir}/final_clustering/best_labels_{reduction_method}_{cluster_method}.csv', 'w') as f:
        labels_df.to_csv(f)
    with open(f'{out_dir}/final_clustering/best_embeddings_{reduction_method}_{cluster_method}.csv', 'w') as f:
        embeddings_df.to_csv(f)

    end_time = time.time()
    final_clustering_time = end_time - start_time
    with open(timing_file, 'a') as f:
        f.write(f"Final Clustering Time: {final_clustering_time:.2f} seconds\n")
    print("Final data clustering completed")
    print("################################# \n\n\n")

    print("Extracting centroids")
    print("################################# \n\n\n")
    start_time = time.time()

    centroid_df = centroids_extraction(pred_path, clusterer, embeddings_df, out_dir)
    with open(f'{out_dir}/final_clustering/best_centroids_{reduction_method}_{cluster_method}.csv', 'w') as f:
        centroid_df.to_csv(f)

    end_time = time.time()
    centroid_extraction_time = end_time - start_time
    with open(timing_file, 'a') as f:
        f.write(f"Centroid Extraction Time: {centroid_extraction_time:.2f} seconds\n")
    print("Centroid extraction completed")
    print("################################# \n\n\n")

    print("GNN Drug-Likeness example model pipeline completed")
    print("################################# \n\n\n")

    print("Trained Model ID:", model_id)
    print("\nExperiment ID:", experiment_id)
    

    smiles_to_inchikey(f'{data_dir}/tm-mc2/preprocessed_TMMC_compounds.csv', f'{out_dir}/dataset_with_inchikey.csv')

    generate_centroids_with_inchikey(f'{out_dir}/centroids_id_molecola.csv', f'{out_dir}/dataset_with_inchikey.csv', f'{out_dir}/centroids_with_inchikey.csv')

    centroids_to_TMMC(f'{out_dir}/centroids_with_inchikey.csv', out_dir, f'{data_dir}/tm-mc2/')


    print("\nModel pipeline execution times:\n")
    with open(timing_file, 'r') as f:
        print(f.read())
    


if __name__ == "__main__":
    
    run_pipeline()
