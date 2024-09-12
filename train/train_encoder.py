import torch
import torch.nn as nn
from utils.graph_processing import process_adj
from utils.operations import save_checkpoint
import os
import torch.nn.functional as F


def train_encoder(G: nn.Module,
                  D: nn.Module,
                  E: nn.Module,
                  e_optimizer: torch.optim.Optimizer,
                  epochs: int,
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  kappa: int,
                  gumbell_type: str,
                  checkpoint_dir: str
                  ):
    
    G.eval()
    D.eval()
    E.train()

    avg_eloss = 0

    for epoch in range(epochs):
        
        total_eloss = 0
        batch_count = 0

        # Discriminator guided gzg_f approach (graph_real, z_hat, graph_fake)
        for batch in data_loader:
            
            adj = batch['adj'].float().clone().to(device)
            x = batch['feat'].float().clone().to(device)

            z_hat = E(x, adj)
            
            # Weights for G and D remain fixed
            adj_logits, x_tilde = G(z_hat)
            adj_tilde = process_adj(adj_logits, gumbell_type)

            _, real_emb = D(x, adj)
            _, fake_emb = D(x_tilde, adj_tilde)

            adj_loss = F.mse_loss(adj, adj_tilde)
            x_loss = F.mse_loss(x, x_tilde)
            guided_dloss = F.mse_loss(real_emb, fake_emb)

            eloss = adj_loss + x_loss + kappa * guided_dloss

            total_eloss += eloss.item()
            batch_count += 1

            # Backward and optimize
            e_optimizer.zero_grad()
            eloss.backward()
            e_optimizer.step()

        avg_eloss = total_eloss / batch_count
        print(f'\rEncoder Training | Epoch {epoch + 1}/{epochs} | Encoder Loss: {avg_eloss:.4f}', end='')

    print()  
    save_checkpoint(E, e_optimizer, epochs, avg_eloss, os.path.join(checkpoint_dir, 'E_checkpoint_final.pth'))