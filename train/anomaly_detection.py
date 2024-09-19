import torch
from utils.graph_processing import process_adj
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
from utils.operations import save_scores_to_file

def detect_anomalies(G, E, D, data_loader, kappa, device, args, epoch=None, file_path='scores.pkl'):
    G.eval()
    E.eval()
    D.eval()

    all_anomaly_scores = []
    all_labels = []

    for batch in data_loader:
        
        adj = batch['adj'].float().clone().to(device)
        x = batch['feat'].float().clone().to(device)
        label = batch['label'].float()

        with torch.no_grad():
            z_hat = E(x, adj)

            adj_logits, x_tilde = G(z_hat)
            adj_tilde = process_adj(adj_logits, args.gumbell_type)
            
            _, real_emb = D(x, adj)
            _, fake_emb = D(x_tilde, adj_tilde)
            
            z_tilde = E(x_tilde, adj_tilde)

        # compute graph distance
        adj_loss_per_graph = F.mse_loss(adj, adj_tilde, reduction='none').mean(dim=(1, 2))  
        x_loss_per_graph = F.mse_loss(x, x_tilde, reduction='none').mean(dim=(1, 2))  
        guided_dloss_per_graph = F.mse_loss(real_emb, fake_emb, reduction='none').mean(dim=1)  
        graph_distance_per_graph = adj_loss_per_graph + x_loss_per_graph + kappa * guided_dloss_per_graph
        
        # compute z distance
        z_distance_per_graph = F.mse_loss(z_hat, z_tilde, reduction='none').mean(dim=1) 
        
        anomaly_score_per_graph = graph_distance_per_graph + z_distance_per_graph

        # store the scores and ground truth labels 
        all_anomaly_scores.extend(anomaly_score_per_graph.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    threshold = np.percentile(all_anomaly_scores, 95)
    predictions = [1 if score > threshold else 0 for score in all_anomaly_scores]  # 1: anomalous, 0: normal

    all_labels = [0 if label == 0 else 1 for label in all_labels]
    
    auc_score = roc_auc_score(all_labels, predictions) 
    
    if epoch is not None:
        if not args.quiet:
            print(f"AUC score at epoch {epoch}: {auc_score}")
        save_scores_to_file(file_path, epoch, auc_score)
    else:
        if not args.quiet:
            print(f"Final AUC score: {auc_score}")
        save_scores_to_file(file_path, 'final', auc_score)
