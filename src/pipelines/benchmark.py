# execute this code to use our pre-trained model on your choice dataset and extract the merics.
# === Determinism bootstrap (PRIME RIGHE del file) ===
import os, random, numpy as np
SEED = 42

# Env vars devono essere settate PRIMA che si importi torch/cuda
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch  # importa torch solo dopo aver settato le env vars
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

import sys

# Path assoluto (non relativo al cwd): il dataset featurizer puo' usare un ProcessPoolExecutor con
# mp_context="spawn", che ricostruisce questo script come "__mp_main__" in ogni processo worker se
# il dataset passato non e' ancora in cache. Un path relativo si romperebbe appena il cwd cambia
# (vedi os.chdir sotto), rompendo gli import in ogni worker.
# Lo script sta in src/pipelines/: la radice del repo e' due livelli sopra.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_REPO_DIR, 'src')
sys.path.append(os.path.join(_SRC_DIR, 'gnn'))


def run_test(test_data_path):
    from predict import predict_drug  # import qui e non a livello di modulo: vedi commento sopra

    # Percorso assoluto prima di cambiare cartella. predict_drug calcola le metriche solo se il
    # CSV ha la colonna 'Drug'; senza, produce soltanto le predizioni.
    test_data_path = os.path.abspath(test_data_path)

    os.chdir(_SRC_DIR)  # path assoluto: idempotente anche se richiamato piu' volte

    model_id = '64582bc220f74718892ff3eb63ba31a0'
    experiment_id = '418303031868899849'
    _, _ = predict_drug(test_data_path, model_id, experiment_id, False, False, True)


if __name__ == "__main__":


    if len(sys.argv) < 2:
        print("Usage: python myscript.py <arg1> [arg2]")
        sys.exit(1)

    test_data_path = sys.argv[1]

    run_test(test_data_path)