# %% imports
import csv
from typing import Dict, Optional
import io
import logging
import os
import sys
import warnings
import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
import mlflow
import pathlib
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
from config import SIGNATURE
from model_architecture.tree_path import myModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.getLogger("deepchem").setLevel(logging.ERROR)
logging.getLogger("deep-chem").setLevel(logging.ERROR)
logging.getLogger("deep_chem").setLevel(logging.ERROR)




warnings.filterwarnings("ignore")

MLRUNS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn,trial = None):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = torch.zeros((), device=device)  # accumulo su GPU: nessuna sync CPU<->GPU per batch
    step = 0
    for batch_index, (batch, tree_batch) in enumerate(tqdm(train_loader)):  #here train_loader is a DataLoader version specific for the graph data 
        #batch is the data for the current batch in the DataLoader
        # Use GPU
        batch.to(device, non_blocking=True)
        tree_batch.to(device, non_blocking=True)
        # Reset gradients
        optimizer.zero_grad() 
        # Estrazione delle features dalle molecole
        data = {
            "mol_raws": {
                "x": batch.x.float(),
                "edge_attr": batch.edge_attr.float(),
                "edge_index": batch.edge_index,
                "batch": batch.batch,
            },
            "mol_trees": {
                "tree_node_features": tree_batch.x.float(),
                "tree_edge_attr":tree_batch.edge_attr.float(),
                "tree_edge_index": tree_batch.edge_index ,
                "tree_batch_index": tree_batch.batch,
            }
        }

      
        # Forward pass
        pred,_,_,_,_ = model(data, device)
        

        # Calculating the loss and gradients
        logits = pred.view(-1)
        targets = batch.y.view(-1).float()
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()
        # Update tracking (resta tutto su GPU, nessuna sync CPU<->GPU per batch)
        running_loss += loss.detach()
        step += 1
        all_preds.append(torch.sigmoid(logits).detach())
        all_labels.append(batch.y.detach())
    # Una sola sincronizzazione GPU->CPU per l'intera epoca, invece che una per batch
    all_preds = np.rint(torch.cat(all_preds).cpu().numpy()).ravel()
    all_labels = torch.cat(all_labels).cpu().numpy().ravel()
    valid_preds = _ensure_valid_predictions(all_preds, stage="train")
    f1,acc,prec,rec,roc  = calculate_metrics(all_preds, all_labels, valid_predictions=valid_preds)
    return (running_loss / step).item()


def test(epoch, model, test_loader, loss_fn,prova_id):
    all_preds = []
    all_labels = []
    running_loss = torch.zeros((), device=device)  # accumulo su GPU: nessuna sync CPU<->GPU per batch
    step = 0
    for batch_index, (batch, tree_batch) in enumerate(tqdm(test_loader)):

        batch.to(device, non_blocking=True)
        tree_batch.to(device, non_blocking=True)

        data = {
            "mol_raws": {
                "x": batch.x.float(),
                "edge_attr": batch.edge_attr.float(),
                "edge_index": batch.edge_index,
                "batch": batch.batch,
            },
            "mol_trees": {
                "tree_node_features": tree_batch.x.float(),
                "tree_edge_attr": tree_batch.edge_attr.float(),
                "tree_edge_index": tree_batch.edge_index,
                "tree_batch_index": tree_batch.batch,
            }
        }


        # Forward pass
        pred,_,_,_,_ = model(data, device)
        logits = pred.view(-1)
        targets = batch.y.view(-1).float()
        loss = loss_fn(logits, targets)

         # Update tracking (resta tutto su GPU, nessuna sync CPU<->GPU per batch)
        running_loss += loss.detach()
        step += 1
        all_preds.append(torch.sigmoid(logits).detach())
        all_labels.append(batch.y.detach())

    # Una sola sincronizzazione GPU->CPU per l'intera epoca, invece che una per batch
    all_preds = np.rint(torch.cat(all_preds).cpu().numpy()).ravel()
    all_labels = torch.cat(all_labels).cpu().numpy().ravel()
    valid_preds = _ensure_valid_predictions(all_preds, stage="test")
    f1,acc,prec,rec,roc  = calculate_metrics(all_preds, all_labels, valid_predictions=valid_preds)
    log_conf_matrix(all_preds, all_labels, epoch,provaid=prova_id, valid_predictions=valid_preds)
    return (running_loss/step).item(), f1,acc,prec,rec,roc

def log_conf_matrix(y_pred, y_true, epoch,provaid, valid_predictions: bool = True):
    if valid_predictions:
        num_classes = len(np.unique(np.concatenate((y_true, y_pred))))
        if num_classes == 2:
            cm = confusion_matrix(y_true, y_pred)
        else:
            cm = np.zeros((2, 2), dtype=int)
    else:
        cm = np.zeros((2, 2), dtype=int)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')

    
    os.makedirs(f'../results/validation/images/conf_matrix/prova_{provaid}', exist_ok=True)
    out = f'../results/validation/images/conf_matrix/prova_{provaid}/cm_{epoch}.png'
    cfm_plot.figure.savefig(out)
    mlflow.log_artifact(out)

def _sanitize_metric(value: float) -> float:
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            return 0.0
        return float(value)
    return 0.0

def _ensure_valid_predictions(preds: np.ndarray, stage: str) -> bool:
    """
    Guard rail to detect NaN/inf predictions without stopping the trial.
    """
    valid = np.all(np.isfinite(preds))
    if not valid:
        print(f"[WARN] {stage} predictions contain non finite values; metrics and confusion matrix will be zeroed.")
    return valid


DEFAULT_METRICS = {"f1": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "roc": 0.0}


def _sanitize_metrics_dict(metrics: Optional[Dict[str, float]]) -> Dict[str, float]:
    clean = DEFAULT_METRICS.copy()
    if not isinstance(metrics, dict):
        return clean
    for key, value in metrics.items():
        if key in clean and isinstance(value, (int, float)) and np.isfinite(value):
            clean[key] = float(value)
    return clean


def calculate_metrics(y_pred, y_true, valid_predictions: bool = True):
    if not valid_predictions:
        f1 = acc = prec = rec = roc = 0.0
        print("\n Confusion matrix: \n [[0 0]\n [0 0]]")
        print("F1 Score: 0.0")
        print("Accuracy: 0.0")
        print("Precision: 0.0")
        print("Recall: 0.0")
        print("ROC AUC: 0.0")
        return f1, acc, prec, rec, roc
    try:
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_pred)
    except Exception:
        f1 = acc = prec = rec = roc = 0.0
        print("There was a problem calculating some of the metrics!")

    f1 = _sanitize_metric(f1)
    acc = _sanitize_metric(acc)
    prec = _sanitize_metric(prec)
    rec = _sanitize_metric(rec)
    roc = _sanitize_metric(roc)

    print(f"\n Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"ROC AUC: {roc}")
    return f1, acc, prec, rec, roc

def run_one_training(params):
    import mlflow  # assicurati che sia importato in cima al file
    params = params[0]
    train_losses = []
    test_losses = []
    id_prova = params["id_prova"]
    trial = params["trial_obj"]

    print(f"Using {torch.cuda.device_count()} GPUs")

    # Esperimento per Optuna/validation
    mlflow.set_experiment("validation")

    # Ensure any previous MLflow run is closed before starting a new one
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Apri un run per questo trial (così log_artifact/log_model funzionano)
    with mlflow.start_run(run_name=f"trial-{id_prova}"):
        # Logga i parametri (escludi oggetti non serializzabili)
        loggable_params = {k: v for k, v in params.items() if k != "trial_obj"}
        for k, v in loggable_params.items():
            mlflow.log_param(k, v)

        # Loading the dataset
        print("Loading dataset...")
        train_dataset = MoleculeDataset(filename=params["train_file"])
        test_dataset  = MoleculeDataset(filename=params["test_file"], test=True)

        # Update the model parameters
        params["model_edge_dim"] = train_dataset[0][0].edge_attr.shape[1]
        print(f"params: {params}")

        # Prepare training
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"],
                                  shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader  = DataLoader(test_dataset,  batch_size=params["batch_size"],
                                  shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

        # Loading the model
        print("Loading model...")
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = myModel(feature_size=train_dataset[0][0].x.shape[1], model_params=model_params)
        print(f"Model parameters: {model_params}")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        # Logga numero parametri del modello
        num_params = count_parameters(model)
        print(f"Number of parameters: {num_params}")
        mlflow.log_param("num_params", num_params)

        # Loss / Optim / Sched
        weight   = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
        loss_fn  = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

        # Training loop
        best_loss = 1000.0
        best_metrics = None
        early_stopping_counter = 0

        for epoch in tqdm(range(params["epochs"])):
            if early_stopping_counter <= 25:  # = x * 5
                model.train()
                loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, trial)
                train_losses.append(loss)
                print(f"Epoch {epoch} | Train Loss {loss}")
                # Se vuoi loggare anche qui:
                # mlflow.log_metric("train_loss", float(loss), step=epoch)

                # Validation
                model.eval()
                if epoch % 5 == 0:
                    loss, f1, acc, prec, rec, roc = test(epoch, model, test_loader, loss_fn, prova_id=id_prova)
                    test_losses.append(loss)
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    # Se vuoi loggare anche qui:
                    # mlflow.log_metric("val_loss", float(loss), step=epoch)

                    # Best model
                    if float(loss) < best_loss:
                        best_loss = float(loss)
                        best_metrics = {"f1": f1, "acc": acc, "prec": prec, "rec": rec, "roc": roc}
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        print("Saving best model...")
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                plot_loss_curves(train_losses, test_losses, id_prova)
                best_metrics = _sanitize_metrics_dict(best_metrics)
                return best_loss, best_metrics

        # Fine training
        plot_loss_curves(train_losses, test_losses, id_prova)
        best_metrics = _sanitize_metrics_dict(best_metrics)
        print(f"Finishing training with best test loss: {best_loss} and best metrics: {best_metrics}")
        return best_loss, best_metrics


def plot_loss_curves(train_losses, val_losses,id_prova):
    os.makedirs("../results/validation/images/losses", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) * 5 + 1, 5), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.savefig(f"../results/validation/images/losses/loss_curves_prova{id_prova}.png")


# %% Optimization with Optuna
# Function to save results to a CSV file.
def save_trial_results(trial_number, params, loss, metrics, filename="../results/validation/optuna_results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header only if file does not exist.
            header = ["trial_number"] + list(params.keys()) + ["loss"] + list(metrics.keys())
            writer.writerow(header)
        
        # Scrivi i risultati della prova
        row = [trial_number] + list(params.values()) + [loss] + list(metrics.values())
        writer.writerow(row)



def objective(trial, training = None, validation = None):
    # Define parameters using the current trial
    if training is not None and validation is not None:
        params = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.1),
        "weight_decay": trial.suggest_loguniform("weight_decay", 0.00001, 0.01),
        "sgd_momentum": trial.suggest_uniform("sgd_momentum", 0.8, 0.99),
        "scheduler_gamma": trial.suggest_uniform("scheduler_gamma", 0.8, 0.99),
        "pos_weight": trial.suggest_categorical("pos_weight", [0.5, 0.7, 0.9, 1.0, 1.3, 1.5]),
        "model_embedding_size": trial.suggest_categorical("model_embedding_size", [64, 128, 256, 512]),
        "model_attention_heads": trial.suggest_categorical("model_attention_heads", [1, 2, 3, 4, 5]),
        "model_layers": trial.suggest_categorical("model_layers", [1, 2, 3, 4, 5]),
        "model_dropout_rate": trial.suggest_uniform("model_dropout_rate", 0.2, 0.8),
        "model_top_k_ratio": trial.suggest_uniform("model_top_k_ratio", 0.25, 0.8),
        "model_top_k_every_n": trial.suggest_categorical("model_top_k_every_n", [1, 2, 3, 5]),
        "model_dense_neurons": trial.suggest_categorical("model_dense_neurons", [128, 256, 512]),
        'epochs': 21,# the correct value is 21 put to five to keep the test short
        'train_file': training,
        'test_file': validation
    }
    else:
        params = {
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.1),
            "weight_decay": trial.suggest_loguniform("weight_decay", 0.00001, 0.01),
            "sgd_momentum": trial.suggest_uniform("sgd_momentum", 0.8, 0.99),
            "scheduler_gamma": trial.suggest_uniform("scheduler_gamma", 0.8, 0.99),
            "pos_weight": trial.suggest_categorical("pos_weight", [0.5, 0.7, 0.9, 1.0, 1.3, 1.5]),
            "model_embedding_size": trial.suggest_categorical("model_embedding_size", [64, 128, 256, 512]),
            "model_attention_heads": trial.suggest_categorical("model_attention_heads", [1, 2, 3, 4, 5]),
            "model_layers": trial.suggest_categorical("model_layers", [1, 2, 3, 4, 5]),
            "model_dropout_rate": trial.suggest_uniform("model_dropout_rate", 0.2, 0.8),
            "model_top_k_ratio": trial.suggest_uniform("model_top_k_ratio", 0.25, 0.8),
            "model_top_k_every_n": trial.suggest_categorical("model_top_k_every_n", [1, 2, 3, 5]),
            "model_dense_neurons": trial.suggest_categorical("model_dense_neurons", [128, 256, 512]),
            'epochs': 21,# lowered from 21 to keep the quick run consistent
            'train_file': "data/zinc/splits/new2_ZINC_final_balanced_train_set.csv",
            'test_file': "data/zinc/splits/new2_ZINC_final_balanced_validation_set.csv"
        }

        
    # print(f"Testing configuration: {params}")
    params["id_prova"] = trial.number
    params["trial_obj"] = trial
    loss, metrics = run_one_training([params])

    roc = metrics["roc"]

    # Save the test results to a CSV file.
    save_trial_results(trial.number, params, loss, metrics)
    return loss, -roc

distributions = {
    "batch_size": optuna.distributions.CategoricalDistribution([16, 32, 64, 128]),
    "learning_rate": optuna.distributions.LogUniformDistribution(0.0001, 0.1),
    "weight_decay": optuna.distributions.LogUniformDistribution(0.00001, 0.01),
    "sgd_momentum": optuna.distributions.UniformDistribution(0.8, 0.99),
    "scheduler_gamma": optuna.distributions.UniformDistribution(0.8, 0.99),
    "pos_weight": optuna.distributions.CategoricalDistribution([0.5, 0.7, 0.9, 1.0, 1.3, 1.5]),
    "model_embedding_size": optuna.distributions.CategoricalDistribution([64, 128, 256, 512]),
    "model_attention_heads": optuna.distributions.CategoricalDistribution([1, 2, 3, 4, 5]),
    "model_layers": optuna.distributions.CategoricalDistribution([1, 2, 3, 4, 5]),
    "model_dropout_rate": optuna.distributions.UniformDistribution(0.2, 0.8),
    "model_top_k_ratio": optuna.distributions.UniformDistribution(0.25, 0.8),
    "model_top_k_every_n": optuna.distributions.CategoricalDistribution([1, 2, 3, 5]),
    "model_dense_neurons": optuna.distributions.CategoricalDistribution([128, 256, 512]),
}

def param_fine_tuning(training = None, validation = None, load_state = False):
    output_dir = "../results/validation"
    os.makedirs(output_dir, exist_ok=True)

    #load_state = False
    n_trials = 150 # the correct value is 150 put to 1 to test
    loaded_trials = 0

    study = optuna.create_study(directions=["minimize", "minimize"], load_if_exists=load_state)
    
    if load_state == True:
        if not os.path.exists(f'../results/validation/optuna_results.csv'):
            print("Attenzione: Il file CSV con i risultati precedenti non esiste.")
        else:
            df = pd.read_csv(f'../results/validation/optuna_results.csv')
            if training is not None and validation is not None:
                df = df[(df["train_file"] == training) & (df["test_file"] == validation)]
                print(f"Filtrati {len(df)} trial precedenti per la coppia train={training}, val={validation}.")
            loaded_trials = len(df)
            # Get the number of current trials in the study
            existing_trials = len(study.trials)
            print(f"Numero di trial esistenti nello studio: {existing_trials}")

            # Add previous results to the study.
            for _, row in df.iterrows():
                trial = optuna.trial.create_trial(
                    params={key: row[key] for key in distributions.keys()},  
                    distributions=distributions,  
                    values=[row['loss'],row["roc"]]   
                )
                study.add_trial(trial)

            # Check if all trials have been added.
            total_trials = len(study.trials)
            print(f"Numero di trial nel CSV caricati: {len(df)}")
            print(f"Numero totale di trial dopo l'aggiunta: {total_trials}")
            
            if total_trials - existing_trials == len(df):
                print("Tutti i trial passati sono stati aggiunti correttamente.")
            else:
                print(f"Attenzione: Non tutti i trial sono stati aggiunti correttamente. Aggiunti: {total_trials - existing_trials} su {len(df)}")

    trials_to_run = n_trials
    if load_state:
        trials_to_run = max(0, n_trials - loaded_trials)

    if training is not None and validation is not None:
        if trials_to_run > 0:
            study.optimize(lambda trial: objective(trial, training, validation), n_trials=trials_to_run, show_progress_bar=True)
        else:
            print("Numero desiderato di trial già raggiunto per questa coppia train/val; salto l'ottimizzazione.")
    else:
        if trials_to_run > 0:
            study.optimize(objective, n_trials=trials_to_run,show_progress_bar=True)
        else:
            print("Numero desiderato di trial già raggiunto; salto l'ottimizzazione.")

     # Best trial based on both metrics.
    best_trial = study.best_trials[0]  # This assumes that the former is the better. You can iterate to find the best according to a specific criterion.

    print("Best trial:")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Stampa i parametri del miglior trial
    print("Best hyperparameters:", best_trial.params)
    print("Best loss:", best_trial.values[0])
    print("Best ROC AUC:", -best_trial.values[1])
    return best_trial

# if __name__ == "__main__":
   
#     best_trial = validation_train()
