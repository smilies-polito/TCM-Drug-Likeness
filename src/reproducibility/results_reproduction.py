# execute this code to use our pre-trained model on your choice dataset and extract the merics.
# === Determinism bootstrap (PRIME RIGHE del file) ===
import os, random, numpy as np
SEED = 42

# Env vars devono essere settate PRIMA che si importi torch/cuda
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# === Riproducibilita' fra macchine diverse ===
# Il modello e i frammenti sono gia' riproducibili ovunque (stesso container), ma lo stadio
# UMAP->HDBSCAN divergeva fra CPU di architettura diversa (Intel Xeon vs AMD Ryzen), dando un
# numero di famiglie diverso. Due cause, entrambe neutralizzate qui:
#
# 1) numba compila JIT per la CPU ospite: istruzioni SIMD diverse (AVX2/AVX-512) cambiano
#    l'ordine delle riduzioni floating-point e quindi l'embedding UMAP. "generic" forza un
#    codegen x86-64 comune a tutte le macchine (piu' lento, ma identico ovunque).
# 2) le librerie BLAS parallelizzano le riduzioni in base al numero di core disponibili
#    (qui 32 vs 48): fissando i thread a 1 il risultato non dipende piu' dalla macchina.
os.environ["NUMBA_CPU_NAME"] = "generic"
os.environ["NUMBA_CPU_FEATURES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import torch  # importa torch solo dopo aver settato le env vars
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
import sys
import ast

# Path assoluto (non relativo al cwd): il dataset featurizer puo' usare un ProcessPoolExecutor con
# mp_context="spawn", che ricostruisce questo script come "__mp_main__" in ogni processo worker se
# il dataset passato non e' ancora in cache. Un path relativo si romperebbe appena il cwd cambia
# (vedi os.chdir sotto), rompendo gli import in ogni worker.
# Lo script sta in src/reproducibility/: la radice del repo e' due livelli sopra. I moduli sono
# divisi in sottocartelle di src/ e si importano per nome, quindi ognuna va in sys.path.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_REPO_DIR, 'src')
for _sub in ('gnn', 'analysis', 'paper', 'explain'):
    sys.path.append(os.path.join(_SRC_DIR, _sub))

def run_test(test_data_path, zinc_test_path='data/zinc/splits/new2_ZINC_final_balanced_test_set.csv'):
    """Riproduce i risultati a partire dal modello pre-addestrato.

    Se `zinc_test_path` e' valorizzato, la valutazione sul test set ZINC15 viene eseguita prima
    dell'applicazione ai composti TCM: produce le metriche di classificazione e la distribuzione
    dei punteggi di riferimento usata nelle figure. Passare None per saltarla (piu' veloce, ma
    niente metriche ne' curva di riferimento nel grafico delle distribuzioni).
    """
    # Import qui e non a livello di modulo: cosi' i worker "spawn" del ProcessPoolExecutor (che
    # rieseguono il modulo come "__mp_main__", non "__main__") non li reimportano inutilmente.
    from predict import predict_drug
    from edgeshaper_step import run_edgeshaper_step
    from apply_clustering_method import apply_clustering, centroids_extraction
    from post_proc_pipeline import crea_fingerprint
    from generate_centroids_with_inchikey import generate_centroids_with_inchikey
    from smiles_conversion import smiles_to_inchikey
    from Centroids_to_TMMC import centroids_to_TMMC
    import make_figures
    import manuscript_statistics

    os.chdir(_SRC_DIR)  # path assoluto: idempotente anche se richiamato piu' volte

    out_dir = '../results'
    data_dir = '../data'
    model_id = '64582bc220f74718892ff3eb63ba31a0'
    experiment_id = '418303031868899849'
    small = False
    benchmark = True

    os.makedirs(out_dir, exist_ok=True)

    # === valutazione sul test set ZINC15
    # E' il regime operativo su cui il modello e' stato addestrato, quindi l'unico stadio in cui
    # si possono calcolare metriche di classificazione (i composti TCM non hanno etichette).
    # Tenerla dentro questo script lo rende autosufficiente: chi clona il repository ottiene
    # metriche, predizioni e figure con un solo comando, senza dover lanciare prima benchmark.py.
    if zinc_test_path:
        print("\nEvaluating the pretrained model on the ZINC15 test set")
        print("################################# \n")
        predict_drug(zinc_test_path, model_id, experiment_id, False, False, benchmark)
        print(f"Metrics saved in {out_dir}/test_data/final_test_metrics.json")

    # === predizione molecole
    print("\nApplying the pretrained model to the TCM compounds")
    print("################################# \n")
    _, pred_path = predict_drug(test_data_path, model_id, experiment_id, False, False, benchmark)

    # === attribuzione dei frammenti con EdgeSHAPer
    # predict_drug restituisce anche un file di frammenti ricavato dai pesi di attenzione: quello
    # non si usa piu'. I frammenti su cui lavora il resto della pipeline vengono da qui, con lo
    # stesso formato di colonne, quindi nulla a valle cambia.
    print("\nStarting EdgeSHAPer fragment attribution")
    print("################################# \n")
    frags_path = run_edgeshaper_step(test_data_path, pred_path, out_dir)


    def extract(key, text):
        for line in text.splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
        return None

    # === lettura parametri clustering dal file di testo
    params_file = "../results/trained_model/best_valid_clustering_params.txt"
    with open(params_file, "r", encoding="utf-8") as f:
        text = f.read()

    reduction_method = extract("Reduction Method", text)
    cluster_method   = extract("Cluster Method", text)
    best_param_str   = extract("Best Parameters", text)

    best_param = ast.literal_eval(best_param_str)

    print("Reduction:", reduction_method)
    print("Clustering:", cluster_method)
    print("Params:", best_param)

    # === creazione fingerprint
    # Il vocabolario dei frammenti si ricava dai frammenti selezionati e non dalla voce
    # "Fragment to Index" del file dei parametri: quella elenca i 634 tipi dell'attribuzione per
    # attenzione, mentre EdgeSHAPer ne seleziona altri, e riusarla farebbe fallire la mappatura
    # delle colonne. I parametri di riduzione e clustering letti qui sopra restano invece quelli
    # validati nel paper, come in src/explain/cluster_published_params.py.
    fp_path = os.path.join(out_dir, 'new_molecule_fingerprints2.csv')
    crea_fingerprint(frags_path, fp_path)

    clustering_dir = os.path.join(out_dir, "final_clustering")
    if not os.path.exists(clustering_dir):
        os.makedirs(clustering_dir)

    # === clustering vero e proprio (niente tuning)
    labels_df, clusterer, embeddings_df = apply_clustering(
        reduction_method,
        cluster_method,
        fp_path,
        clustering_dir,
        best_param
    )

    centroid_df = centroids_extraction(pred_path, clusterer, embeddings_df, out_dir)

    smiles_to_inchikey(f'{data_dir}/tm-mc2/preprocessed_TMMC_compounds.csv', f'{out_dir}/dataset_with_inchikey.csv')

    generate_centroids_with_inchikey(f'{out_dir}/centroids_id_molecola.csv', f'{out_dir}/dataset_with_inchikey.csv', f'{out_dir}/centroids_with_inchikey.csv')

    centroids_to_TMMC(f'{out_dir}/centroids_with_inchikey.csv', out_dir, f'{data_dir}/tm-mc2/')

    # === figure del manoscritto ===
    # La distribuzione dei punteggi TCM viene sovrapposta a quella di riferimento sul test set
    # ZINC, prodotta dalla valutazione qui sopra. Se quella fase e' stata saltata il file puo'
    # mancare o riferirsi a un altro dataset (benchmark.py lo sovrascrive): in tal caso la
    # figura viene disegnata con il solo set TCM.
    zinc_predictions = f'{out_dir}/test_data/predictions_raw.csv'
    try:
        make_figures.generate_all(
            out_dir=out_dir,
            reference_predictions_path=zinc_predictions if os.path.exists(zinc_predictions) else None,
        )
    except Exception as e:  # una figura non deve mai invalidare una run di analisi completa
        print(f"[figures] generazione fallita ({e}); i risultati numerici sono comunque salvati")

    # === statistiche citate nel manoscritto ===
    # Ricalcolate dagli artefatti appena salvati, cosi' ogni numero che compare nel testo ha un
    # punto unico e riproducibile in cui viene prodotto.
    try:
        manuscript_statistics.compute_all(out_dir=out_dir)
    except Exception as e:  # come sopra: una statistica non invalida una run di analisi
        print(f"[stats] calcolo fallito ({e}); i risultati numerici sono comunque salvati")

    

    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Reproduce the published results with the pretrained model: evaluation on '
                    'the ZINC15 test set, cross-domain application to the TCM compounds, '
                    'fragment-level clustering and manuscript figures.')
    parser.add_argument('--compounds',
                        default='data/tm-mc2/preprocessed_TMMC_compounds.csv',
                        help='Compound library to apply the model to (csv with a smiles column)')
    parser.add_argument('--skip-zinc', action='store_true',
                        help='Skip the ZINC15 evaluation. Faster, but no classification metrics '
                             'and no reference distribution in the score-distribution figure')
    args = parser.parse_args()

    run_test(args.compounds,
             zinc_test_path=None if args.skip_zinc
             else 'data/zinc/splits/new2_ZINC_final_balanced_test_set.csv')