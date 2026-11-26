#%% imports 
import os
import time
import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
import mlflow.pytorch
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from model_architecture.tree_building import DGLMolTree
from model_architecture.tree_path import myModel
import warnings
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

import csv
import os
def gpu_check(model, data, interval=10):
    """
    Performs GPU usage checks before and during training.

    Args:
        model: The PyTorch model to be trained.
        data: The data used for training, which must be on the GPU.
        interval: The time interval (in seconds) between each check during training.
    """

    # Check if the GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move the model to the device
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Function to monitor GPU usage during training
    def monitor_gpu_usage():
        while True:
            print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")
            print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")
            time.sleep(interval)

    # Start monitoring GPU usage in a separate thread
    import threading
    monitor_thread = threading.Thread(target=monitor_gpu_usage)
    monitor_thread.daemon = True
    monitor_thread.start()

    return device


def get_next_run_id(file_path):
    if not os.path.isfile(file_path):
        return 0
    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header
        ids = [int(row[0]) for row in reader if row]
    
    if not ids:
        return 0
    return max(ids) + 1


run_id = get_next_run_id('../model_results/training_results.csv')

def save_results(file_path, params, best_train_loss, best_test_loss):
    
    file_exists = os.path.isfile(file_path)
    if not file_exists:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header only if the file does not exist
            header = ["run_id"]+ list(params.keys()) + ['best_train_loss', 'best_test_loss']
            writer.writerow(header)
        row = [run_id]+ list(params.values()) + [best_train_loss, best_test_loss]
        writer.writerow(row)





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = torch.zeros((), device=device)  # accumulo su GPU: nessuna sync CPU<->GPU per batch
    step = 0
    for batch_index, (batch, tree_batch) in enumerate(tqdm(train_loader)):  # here train_loader is a DataLoader version specific for the graph data
        # DEBUG-TIMING: isola il tempo di trasferimento GPU dal tempo di attesa del DataLoader
        _t_transfer0 = time.perf_counter()
        # Use GPU
        batch.to(device, non_blocking=True)
        tree_batch.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # necessario: senza sync .to(device) e' asincrono e il timer non misura nulla di reale
        if batch_index < 3:
            print(f"[DEBUG-TIMING] epoch {epoch} batch {batch_index}: batch+tree .to(device) = {time.perf_counter()-_t_transfer0:.4f}s")
        # Reset gradients
        optimizer.zero_grad() 
    
        # Extracting features from the molecules
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
        pred, _, _, _, _ = model(data, device)
        

        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        # Update tracking (resta tutto su GPU, nessuna sync CPU<->GPU per batch)
        running_loss += loss.detach()
        step += 1
        all_preds.append(torch.sigmoid(pred).detach())
        all_labels.append(batch.y.detach())
    # Una sola sincronizzazione GPU->CPU per l'intera epoca, invece che una per batch
    all_preds = np.rint(torch.cat(all_preds).cpu().numpy()).ravel()
    all_labels = torch.cat(all_labels).cpu().numpy().ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return (running_loss / step).item()

def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = torch.zeros((), device=device)  # accumulo su GPU: nessuna sync CPU<->GPU per batch
    step = 0
    for batch_index, (batch, tree_batch) in enumerate(tqdm(test_loader)):
        # Now you have access to `batch` for molecules and `tree_batch` for trees.

        batch.to(device, non_blocking=True)
        tree_batch.to(device, non_blocking=True)
        
        # Extracting features from the molecules
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
        pred, _, _, _, _ = model(data, device)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())

         # Update tracking (resta tutto su GPU, nessuna sync CPU<->GPU per batch)
        running_loss += loss.detach()
        step += 1
        all_preds.append(torch.sigmoid(pred).detach())
        all_labels.append(batch.y.detach())

    # Una sola sincronizzazione GPU->CPU per l'intera epoca, invece che una per batch
    all_preds = np.rint(torch.cat(all_preds).cpu().numpy()).ravel()
    all_labels = torch.cat(all_labels).cpu().numpy().ravel()

    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return (running_loss / step).item()

def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    os.makedirs("../results/images", exist_ok=True)
    cfm_plot.figure.savefig(f'../results/images/cm_{epoch}.png')
    mlflow.log_artifact(f"../results/images/cm_{epoch}.png")

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")



#%% Run the training
from config import SIGNATURE

def run_one_training(params):
    params = params[0]
    train_losses = []
    test_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_experiment("0")
    # Ensure no stray MLflow run is still active (e.g., from previous trials)
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run() as run:
        
        # Log parameters used in this experiment
        for key in params.keys():
            mlflow.log_param(key, params[key])

        # Loading the dataset
        print("Loading dataset...")
        _t_ds0 = time.perf_counter()  # DEBUG-TIMING
        train_dataset = MoleculeDataset(filename=params["train_file"])
        test_dataset = MoleculeDataset(filename=params["test_file"], test=True)
        print(f"[DEBUG-TIMING] dataset construction (train+test): {time.perf_counter()-_t_ds0:.2f}s")  # DEBUG-TIMING

        params["model_edge_dim"] = train_dataset[0][0].edge_attr.shape[1]  # train_dataset[0] is the first molecule in the dataset
        print(f"params: {params}")
        # Prepare training
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

        # Loading the model
        print("Loading model...")
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = myModel(feature_size=train_dataset[0][0].x.shape[1], model_params=model_params)
        print(f"Model parameters: {model_params}")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        # < 1 increases precision, > 1 recall
        weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])
        
        # Start training
        best_loss = 1000
        early_stopping_counter = 0
        
        for epoch in tqdm(range(params["epochs"])):
            if early_stopping_counter <= 25:  # = x * 5 
                # Training
                model.train()

                _t_epoch0 = time.perf_counter()  # DEBUG-TIMING
                loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                print(f"[DEBUG-TIMING] epoch {epoch} total train_one_epoch time: {time.perf_counter()-_t_epoch0:.2f}s")  # DEBUG-TIMING

                train_losses.append(loss)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

                # Testing
                model.eval()
                if epoch % 5 == 0:

                    loss = test(epoch, model, test_loader, loss_fn)
                    test_losses.append(loss)
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
                    
                    # Update best loss
                    if float(loss) < best_loss:
                        best_loss = loss
                        # Save the currently best model 
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        print("Saving best model...")
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                plot_loss_curves(train_losses, test_losses)
                model_id = run.info.run_id
                experiment_id = run.info.experiment_id
                return best_loss, model_id, experiment_id
    plot_loss_curves(train_losses, test_losses)
    print(f"Finishing training with best test loss: {best_loss}")
    save_results('../model_results/training_results.csv', params, min(train_losses), min(test_losses))
    model_id = run.info.run_id
    experiment_id = run.info.experiment_id
    return best_loss, model_id, experiment_id

def plot_loss_curves(train_losses, test_losses):
    # Plotting the loss curves
    os.makedirs("../results/images", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(test_losses) * 5 + 1, 5), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss Curves')
    plt.savefig(f"../results/images/loss_curves_config_{run_id}.png")
    mlflow.log_artifact(f"../results/images/loss_curves_config_{run_id}.png")
    
    plt.show()
    

def training(params=None, epochs=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Specify tracking server
    mlflow.set_tracking_uri("../results/mlruns")
    
    
    config = dict()
   
    BEST_PARAMETERS = {
        'batch_size': 16,
        'learning_rate': 0.0012616942661620749,
        'weight_decay': 0.0010531753345669505, 
        'sgd_momentum': 0.9229827070466566, 
        'scheduler_gamma': 0.8616991325330241, 
        'pos_weight': 0.5, 
        'model_embedding_size': 64, 
        'model_attention_heads': 3, 
        'model_layers': 4,
        'model_dropout_rate': 0.29090904686145314, 
        'model_top_k_ratio': 0.6163542509531891, 
        'model_top_k_every_n': 2, 
        'model_dense_neurons': 128,
        'epochs': epochs,
        'train_file': "data/zinc/splits/new2_ZINC_final_balanced_train_set.csv",
        'test_file': "data/zinc/splits/new2_ZINC_final_balanced_validation_set.csv"
    }

    # Create config using BEST_PARAMETERS
    if params is None:
        for key in BEST_PARAMETERS.keys():
            config[key] = BEST_PARAMETERS[key]
    else:
        config = {
            'epochs': epochs,
            'train_file': "data/zinc/splits/new2_ZINC_final_balanced_train_set.csv",
            'test_file': "data/zinc/splits/new2_ZINC_final_balanced_validation_set.csv"
        }
        for key in params.keys():
            config[key] = params[key]

    running_loss, model_id, exp_id = run_one_training([config])

    return running_loss, model_id, exp_id



# # Create a dummy model
# if __name__ == "__main__":
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Specify tracking server
#     mlflow.set_tracking_uri("http://localhost:5000")
  
#     # %% Hyperparameter search
#     # print("Running hyperparameter search...")
#     config = dict()
#     # config["optimizer"] = "Bayesian"

#     # This config also seems to be good
#     # params={'batch_size': 16, 'learning_rate': 0.0014529236861494325, 'weight_decay': 7.23152103258082e-05, 'sgd_momentum': 0.8946112505430376, 'scheduler_gamma': 0.8805917002592172, 'pos_weight': 1.3, 'model_embedding_size': 128, 'model_attention_heads': 4, 'model_layers': 1, 'model_dropout_rate': 0.2658137351601765, 'model_top_k_ratio': 0.745862388949835, 'model_top_k_every_n': 3, 'model_dense_neurons': 512, 'epochs': 20, 'train_file': 'medium_new2_ZINC_final_balanced_train_set.csv', 'test_file': 'small_new2_ZINC_final_balanced_validation_set.csv', 'model_edge_dim': 11}
# # Model parameters: {'model_embedding_size': 128, 'model_attention_heads': 4, 'model_layers': 1, 'model_dropout_rate': 0.2658137351601765, 'model_top_k_ratio': 0.745862388949835, 'model_top_k_every_n': 3, 'model_dense_neurons': 512, 'model_edge_dim': 11}

#     # BEST_PARAMETERS = {
#     # 'batch_size': 64,  # previously it was 64!!!!!!!!!!
#     # 'learning_rate': 0.014015474347793431, 
#     # 'weight_decay': 0.002449693011254293, 
#     # 'sgd_momentum': 0.8357800974603684, 
#     # 'scheduler_gamma': 0.972996507445344, 
#     # 'pos_weight': 1.0, 
#     # 'model_embedding_size': 256, 
#     # 'model_attention_heads': 2, 
#     # 'model_layers': 2, 
#     # 'model_dropout_rate': 0.20040422634492724, 
#     # 'model_top_k_ratio': 0.5995492263146199, 
#     # 'model_top_k_every_n': 1, 
#     # 'model_dense_neurons': 128,
#     #     'epochs': 201,
#     #     'train_file': "new2_ZINC_final_balanced_train_set.csv",
#     #     'test_file': "new2_ZINC_final_balanced_validation_set.csv"
#     # }

#     BEST_PARAMETERS = {
#         'batch_size': 32, 
#         'learning_rate': 0.0011862515762923046, 
#         'weight_decay': 0.0005073072978716553, 
#         'sgd_momentum': 0.8295220628523434, 
#         'scheduler_gamma': 0.9059721848106833, 
#         'pos_weight': 0.7, 
#         'model_embedding_size': 256, 
#         'model_attention_heads': 3, 
#         'model_layers': 1,
#         'model_dropout_rate': 0.22277792420399223, 
#         'model_top_k_ratio': 0.28771088919741555, 
#         'model_top_k_every_n': 1, 
#         'model_dense_neurons': 256,
#         'epochs': 701,
#         'train_file': "new2_ZINC_final_balanced_train_set.csv",
#         'test_file': "new2_ZINC_final_balanced_validation_set.csv"
#     }

    
#     # Create config using BEST_PARAMETERS
#     for key in BEST_PARAMETERS.keys():
#         config[key] = BEST_PARAMETERS[key]

#     running_loss = run_one_training([config])
#     # tuner = Tuner(BEST_PARAMETERS, 
#     #               objective=run_one_training,
#     #               conf_dict=config) 
#     # results = tuner.minimize()

    # training()
