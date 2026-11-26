import argparse
import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader  
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from model_architecture.highlight_mol import highlight_mol_per_graph
from IPython.display import display

from PIL import Image
import io
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
matplotlib.use('Agg')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(filepath):
    print(f"Loading model from {filepath}")
    model = torch.load(filepath, map_location=device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
    model.to(device)
    print(f"Model loaded successfully.")
    return model

def calculate_metrics(y_pred, y_true, epoch, split, final, probs=None):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    try:
        # roc_auc_score richiede uno score continuo, non la predizione gia' binarizzata:
        # con un input binario collassa sulla balanced accuracy invece di essere un vero ROC-AUC.
        roc = roc_auc_score(y_true, probs if probs is not None else y_pred)
    except:
        roc = "not defined"
    
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"ROC AUC: {roc}")

    metrics = {
        "Confusion Matrix": confusion_matrix(y_pred, y_true).tolist(),
        "F1 Score": f1,
        "Accuracy": accuracy,
        "Precision": prec,
        "Recall": rec,
        "ROC AUC": roc
    }

    if final:
        metrics_path = '../results/test_data/TCMBANK/final_test_metrics.json'
    else:
        metrics_path = '../results/test_data/final_test_metrics.json'

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

def log_conf_matrix(y_pred, y_true, epoch,final = False):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    if final == True:
        os.makedirs('../results/test_data/TCMBANK/images', exist_ok=True)
        cfm_plot.figure.savefig(f'../results/test_data/TCMBANK/images/cm_final_test.png')
    else:
        os.makedirs('../results/test_data/images', exist_ok=True)
        cfm_plot.figure.savefig(f'../results/test_data/images/cm_final_test.png')

def optimize_threshold(pred_probs, true_labels):
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        preds = (pred_probs >= threshold).astype(int)
        f1 = f1_score(true_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def test_model(model, test_data_path, batch_size, pos_weight, deep_search=True, final=False, compute_metrics=True, return_details=False):
    # Load the test dataset
    if final == True:
        test_dataset = MoleculeDataset(filename=test_data_path, test=True,final=final)
    else:
        if not os.path.exists(test_data_path):
            test_data_path = test_data_path.replace('../', '')
        test_dataset = MoleculeDataset(filename=test_data_path, test=True,final=final)
        print(f"Number of test samples: {len(test_dataset)}")
    g = torch.Generator()
    g.manual_seed(42)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,     # mantiene l'ordine
        num_workers=0,     # 0 = ordine deterministico
        pin_memory=False,  # opzionale; riduce variabilità
        generator=g        # seed esplicito
    )

    # Define the loss function
    weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    
    # Run testing
    model.eval()
    all_preds = []
    all_labels = []
    all_smiles = []
    running_loss = 0.0
    step = 0
    high_attention_frags = []
    with torch.no_grad():
        prev_batch_num_graphs = 0
        for batch_index, (batch, tree_batch) in enumerate(tqdm(test_loader)):  #here train_loader is a DataLoader version specific for the graph data 
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
                "tree_edge_attr":tree_batch.edge_attr.float(),
                "tree_edge_index": tree_batch.edge_index ,
                "tree_batch_index": tree_batch.batch,
            }
        }
            trees = tree_batch.mol_tree
            pred,raw_edge_index,raw_attention_scores,tree_edge_index,tree_attention_scores = model(data,device)

            num_graphs,high_score_fragments = highlight_mol_per_graph(raw_edge_index, raw_attention_scores, tree_edge_index,tree_attention_scores, batch,tree_batch,prev_batch_num_graphs,deep_search=deep_search,final=final)
            
            prev_batch_num_graphs += num_graphs
            high_attention_frags.extend(high_score_fragments)
            
            if final == False:
                loss = loss_fn(torch.squeeze(pred), batch.y.float())
                running_loss += loss.item()
                all_labels.append(batch.y.cpu().detach().numpy())

            step += 1
            all_preds.append(torch.sigmoid(pred).cpu().detach().numpy())
            # save SMILES and prediction for each molecule in the batch.
            for i in range(len(batch.smiles)):
                all_smiles.append(batch.smiles[i])
            

            

    all_preds = np.concatenate(all_preds).ravel()
    if final == False:
        all_labels = np.concatenate(all_labels).ravel()
    print(f"Predictions: {all_preds}")

    if final == False:
        print(f"Labels: {all_labels}")
        test_loss = running_loss / step
    else:
        test_loss = None

    

    # Calculate evaluation metrics.
    if final == False:
        best_threshold = 0.5
        final_preds = (all_preds >= best_threshold).astype(int)
        if compute_metrics:
            print(f"Best threshold: {best_threshold}")
            calculate_metrics(final_preds, all_labels, 0, "test", final=final, probs=all_preds)
            log_conf_matrix(final_preds, all_labels, 0, final=final)

            raw_preds_df = pd.DataFrame({
                "smiles": all_smiles,
                "label": all_labels,
                "probability": all_preds,
                "pred": final_preds,
            })
            raw_preds_path = '../results/test_data/predictions_raw.csv'
            raw_preds_df.to_csv(raw_preds_path, index=False)
            print(f"Raw predictions saved to {raw_preds_path}")
    else:
        final_preds = (all_preds >= 0.5).astype(int)
        file_name = "TMMC_molecule_predictions.csv"
        predictions_df = pd.DataFrame({"smiles": all_smiles, "pred": final_preds})
        predictions_df = predictions_df.sort_values(["smiles"]).reset_index(drop=True)  # 👈 ordina per chiave stabile

        os.makedirs('../results/test_data/TCMBANK', exist_ok=True)
        predictions_df.to_csv(f"../results/test_data/TCMBANK/{file_name}", index=False)
        prediction_path = f"../results/test_data/TCMBANK/{file_name}"
        print(f"Predictions saved in {prediction_path}")

        # Oltre alla predizione binaria si salva anche la probabilita' continua: serve per
        # analizzare la distribuzione dei punteggi sul set TCM (e confrontarla con quella
        # ottenuta sul test set ZINC), informazione che andrebbe altrimenti persa.
        raw_preds_df = pd.DataFrame({"smiles": all_smiles, "probability": all_preds,
                                     "pred": final_preds})
        raw_preds_df = raw_preds_df.sort_values(["smiles"]).reset_index(drop=True)
        raw_preds_path = "../results/test_data/TCMBANK/predictions_raw.csv"
        raw_preds_df.to_csv(raw_preds_path, index=False)
        print(f"Raw predictions (with probabilities) saved in {raw_preds_path}")
        



    # Save fragments with higher attention
    if final == False:
        high_attention_frags = [
        (smiles, num_neigh, id_molecola, final_preds[id_molecola],all_labels[id_molecola])
        for smiles, num_neigh, id_molecola in high_attention_frags
]
        high_attention_frags_df = pd.DataFrame(high_attention_frags, columns=["smiles","num_neigh","id_molecola","pred","real_label"])
    else:
        high_attention_frags = [
        (smiles, num_neigh, id_molecola, final_preds[id_molecola])
        for smiles, num_neigh, id_molecola in high_attention_frags
]
        high_attention_frags_df = pd.DataFrame(high_attention_frags, columns=["smiles","num_neigh","id_molecola","pred"])

    if "id_molecola" in high_attention_frags_df.columns:
        high_attention_frags_df["id_molecola"] = high_attention_frags_df["id_molecola"].astype(int)

    high_attention_frags_df = high_attention_frags_df.sort_values(
        ["id_molecola", "smiles", "num_neigh"]
    ).reset_index(drop=True)
        

    

    if deep_search == True:
        file_name = "deep_search_high_attention_frags.csv"
    else:
        file_name = "high_attention_frags.csv"

    frags_path = None
    if final == True:
        os.makedirs('../results/test_data/TCMBANK', exist_ok=True)
        high_attention_frags_df.to_csv(f"../results/test_data/TCMBANK/{file_name}", index=False)
        frags_path = f"../results/test_data/TCMBANK/{file_name}"
    else:
        os.makedirs('../results/test_data', exist_ok=True)
        high_attention_frags_df.to_csv(f"../results/test_data/{file_name}", index=False)
        frags_path = f"../results/test_data/{file_name}"    


    details = {
        "probabilities": all_preds,
        "final_predictions": final_preds,
        "smiles": all_smiles,
        "frags_path": frags_path,
    }
    if final == False:
        details["labels"] = all_labels
        details["loss"] = test_loss
    else:
        details["prediction_path"] = prediction_path

    if final == False:
        print(f"Test Loss: {test_loss}")
        base_return = (test_loss, frags_path, None)
    else:
        base_return = (None, frags_path, prediction_path)

    if return_details:
        return base_return + (details,)
    return base_return


def load_config(model_name,experiment_id, benchmark=False):

    print(f"load_config Model name: {model_name}")
    print(f"load_config Experiment ID: {experiment_id}")
    if benchmark == True:
        model_path = os.path.join("../results/trained_model/mlruns", f"{experiment_id}")
    else:
        model_path = os.path.join("../results/mlruns", f"{experiment_id}")
    print(f"load_config Model path: {model_path}")
    run_folders = os.listdir(model_path)
    matching_folder = None

    for folder in run_folders:
        if folder.startswith(model_name):
            matching_folder = folder
            break

    if matching_folder is not None:
        params_folder = os.path.join(model_path, matching_folder, "params")
        params = {}
        for file_name in os.listdir(params_folder):
            file_path = os.path.join(params_folder, file_name)
            with open(file_path, "r") as f:
                param_name = file_name.split(".")[0]
                param_value = f.read()
                params[param_name] = param_value
    
        return params
    else:
        assert matching_folder is not None, "Matching folder not found."
        return None

    

def _has_labels(test_data_path):
    """Un dataset e' etichettato se il CSV ha la colonna 'Drug': in quel caso si calcolano le
    metriche, altrimenti si producono solo le predizioni (composti TCM). I percorsi relativi sono
    rispetto alla radice del repo, come in MoleculeDataset (che li unisce a '../')."""
    return 'Drug' in pd.read_csv(os.path.join('..', test_data_path), nrows=0).columns


def predict_drug(test_data_path, model_id = None,experiment_id = None,deep_search=False, small=False, benchmark = False):    

    final = not _has_labels(test_data_path)
    print(f"Deep search: {deep_search}")
    print(f"Model ID: {model_id}")
    print(f"Experiment ID: {experiment_id}")

    # Determine the feature_size (optional, if needed).
    # train_dataset = MoleculeDataset(root="data", filename=params["train_file"]) 
    # feature_size = train_dataset[0].x.shape[1]
    feature_size = 30  
    if benchmark == True:
        checkpoint_path = f'../results/trained_model/mlartifacts/{experiment_id}/{model_id}/artifacts/model/data/model.pth'
    else:
        checkpoint_path = f'../results/mlruns/{experiment_id}/{model_id}/artifacts/model/data/model.pth'
    params = load_config(model_id,experiment_id, benchmark = benchmark)
    print(f"Loaded params: {params}")

    # Load the saved model as an instance
    model = load_model(checkpoint_path)

    print("Creating folders to save images")
    if final:
        images_path = "../results/test_data/TCMBANK/images"
        tree_images_path = "../results/test_data/TCMBANK/images/tree_decomp"
        raw_images_path = "../results/test_data/TCMBANK/images/mol_raw"
    else:
        images_path = "../results/test_data/images"
        tree_images_path = "../results/test_data/images/tree_decomp"
        raw_images_path = "../results/test_data/images/mol_raw"
    print(images_path)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(tree_images_path):
        os.makedirs(tree_images_path)
    if not os.path.exists(raw_images_path):
        os.makedirs(raw_images_path)


    batch_size = int(params['batch_size'])
    pos_weight = float(params['pos_weight'])
    test_loss,frags_path,pred_path = test_model(model, test_data_path, batch_size, pos_weight,deep_search=deep_search,final=final)

    if final == False:
        print(f"Finished testing with test loss: {test_loss}")
    
    return frags_path,pred_path


def predict_drug_k_fold(test_data_path, experiment_ids, model_ids, deep_search=False, small=False, benchmark=False, output_tag=None):
    """
    Run inference with multiple folds and average the predicted probabilities.
    """
    if len(experiment_ids) != len(model_ids):
        raise ValueError("experiment_ids and model_ids must have the same length.")

    final = not _has_labels(test_data_path)
    probability_runs = []
    labels = None
    smiles = None

    for exp_id, model_id in zip(experiment_ids, model_ids):
        if benchmark:
            checkpoint_path = f'../results/trained_model/mlartifacts/{exp_id}/{model_id}/artifacts/model/data/model.pth'
        else:
            checkpoint_path = f'../results/mlruns/{exp_id}/{model_id}/artifacts/model/data/model.pth'

        params = load_config(model_id, exp_id, benchmark=benchmark)
        if params is None:
            raise ValueError(f"Parameters not found for experiment {exp_id}, model {model_id}.")

        model = load_model(checkpoint_path)
        batch_size = int(float(params['batch_size']))
        pos_weight = float(params['pos_weight'])

        _, _, _, details = test_model(
            model,
            test_data_path,
            batch_size,
            pos_weight,
            deep_search=deep_search,
            final=final,
            compute_metrics=False,
            return_details=True,
        )

        probability_runs.append(details["probabilities"])

        if smiles is None:
            smiles = details["smiles"]
        if not final:
            current_labels = details.get("labels")
            if current_labels is None:
                raise ValueError("Labels are required for evaluation but were not found.")
            if labels is None:
                labels = current_labels
            elif len(labels) != len(current_labels):
                raise ValueError("Mismatch in number of samples between folds.")

    if not probability_runs:
        raise ValueError("No predictions were generated.")

    avg_probs = np.mean(np.vstack(probability_runs), axis=0)
    final_preds = (avg_probs >= 0.5).astype(int)

    def _tagged_path(path):
        if not output_tag:
            return path
        base, ext = os.path.splitext(path)
        return f"{base}_{output_tag}{ext}"

    if final:
        predictions_df = pd.DataFrame(
            {"smiles": smiles, "prob": avg_probs, "pred": final_preds}
        ).sort_values("smiles").reset_index(drop=True)
        os.makedirs('../results/test_data/TCMBANK', exist_ok=True)
        predictions_path = _tagged_path("../results/test_data/TCMBANK/k_fold_ensemble_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Ensemble predictions saved to {predictions_path}")
        return predictions_path

    conf = confusion_matrix(labels, final_preds)
    f1 = f1_score(labels, final_preds)
    acc = accuracy_score(labels, final_preds)
    prec = precision_score(labels, final_preds)
    rec = recall_score(labels, final_preds)
    roc = roc_auc_score(labels, avg_probs)

    metrics = {
        "Confusion Matrix": conf.tolist(),
        "F1 Score": float(f1),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "ROC AUC": float(roc),
    }

    os.makedirs('../results/test_data', exist_ok=True)
    metrics_path = _tagged_path("../results/test_data/k_fold_ensemble_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Ensemble metrics saved to {metrics_path}")

    os.makedirs('../results/test_data/images', exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf, annot=True, cmap='Blues', fmt='g', cbar=False,
                xticklabels=["0", "1"], yticklabels=["0", "1"])
    cm_path = _tagged_path("../results/test_data/images/cm_k_fold_ensemble.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Ensemble confusion matrix saved to {cm_path}")

    predictions_df = pd.DataFrame(
        {"smiles": smiles, "prob": avg_probs, "pred": final_preds, "label": labels}
    ).sort_values("smiles").reset_index(drop=True)
    predictions_path = _tagged_path("../results/test_data/k_fold_ensemble_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Ensemble predictions saved to {predictions_path}")

    return metrics_path, predictions_path
# if __name__ == "__main__":
#     predict_drug()
