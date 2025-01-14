import numpy as np
import torch
import torch.nn as nn
from utils.graph_processing import process_adj
from utils.operations import sample_z, gradient_penalty, update_lr, save_checkpoint, load_checkpoint
import os
from train.train_encoder import train_encoder
import argparse
from train.anomaly_detection import detect_anomalies
import copy
from utils.logs import log_graph

def train_wgan(G: nn.Module,
               D: nn.Module,
               E: nn.Module,
               g_optimizer: torch.optim.Optimizer,
               d_optimizer: torch.optim.Optimizer,
               e_optimizer: torch.optim.Optimizer,
               start_epoch: int,
               args: argparse.Namespace,
               data_loader_train: torch.utils.data.DataLoader,
               data_loader_test: torch.utils.data.DataLoader,
               device: torch.device,
               z_dim: int,
               save_iters: list,
               checkpoint_dir: str,
               lambda_gp: float,
               kappa: float
               ):

    torch.autograd.set_detect_anomaly(True)
    
    losses = {'G': [], 'D': []}

    best_gloss = np.inf
    epochs_no_improve = 0

    avg_dloss = 0
    avg_gloss = 0

    for epoch in range(start_epoch, args.wgan_epochs):

        G.train()
        D.train()
        
        total_dloss = 0
        total_gloss = 0
        batch_count = 0

        # min-max game
        for batch in data_loader_train:
            
            adj = batch['adj'].float().clone().to(device)
            x = batch['feat'].float().clone().to(device)

            if torch.isnan(x).any():
                print(x)
                assert not torch.isnan(x).any(), "Nan in x :("

            current_batch_size = adj.shape[0]
            z = sample_z(current_batch_size, z_dim).to(device)

            # discriminator 
            dloss = train_d_step(G, D, g_optimizer, d_optimizer, x, adj, z, args.gumbell_type, device, lambda_gp)

            # generator 
            if epoch % args.n_critic == 0:
                gloss = train_g_step(G, D, g_optimizer, d_optimizer, z, args.gumbell_type, epoch, dloss)
                total_gloss += gloss.item()

            total_dloss += dloss.item()    
            batch_count += 1

        avg_dloss = total_dloss / batch_count
        if epoch % args.n_critic == 0:
            avg_gloss = total_gloss / batch_count
        
        # print progress
        if not args.quiet:
            print(f'\rWGAN Training | Epoch {epoch + 1}/{args.wgan_epochs} | Generator Loss: {avg_gloss:.4f} | Discriminator Loss: {avg_dloss:.4f}', end='')

        # logging losses
        losses['G'].append(avg_gloss)
        losses['D'].append(avg_gloss)

        # early stopping
        if args.early_stop:
            if avg_gloss < best_gloss:
                best_gloss = avg_gloss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                if not args.quiet:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs due to no improvement in generator loss.", end='')
                break

        # decay learning rate
        if (epoch + 1) % args.lr_update_step == 0 and (epoch + 1) > (args.wgan_epochs - args.epochs_decay):
            args.g_lr -= args.g_lr / float(args.epochs_decay)
            args.d_lr -= args.d_lr / float(args.epochs_decay)
            update_lr(g_optimizer, d_optimizer, args.g_lr, args.d_lr)

        # save checkpoint
        if (epoch + 1) in save_iters: 
            save_checkpoint(G, g_optimizer, epoch + 1, avg_gloss, os.path.join(checkpoint_dir, 'G_checkpoint_{}.pth'.format(epoch + 1)))
            save_checkpoint(D, d_optimizer, epoch + 1, avg_dloss, os.path.join(checkpoint_dir, 'D_checkpoint_{}.pth'.format(epoch + 1)))

            # saving models' current state
            E_initial_state = copy.deepcopy(E.state_dict())
            e_optimizer_initial_state = copy.deepcopy(e_optimizer.state_dict())
            G_current_state = copy.deepcopy(G.state_dict())
            D_current_syate = copy.deepcopy(D.state_dict())

            # testing
            print()
            train_encoder(G, D, E, e_optimizer, args.encoder_epochs, data_loader_train, device, kappa, args.gumbell_type, checkpoint_dir, args)
            _, _ = load_checkpoint(E, e_optimizer, os.path.join(checkpoint_dir, 'E_checkpoint_final.pth'))
            detect_anomalies(G, E, D, data_loader_test, kappa, device, args, epoch+1)

            # rolling back to states before E training  
            E.load_state_dict(E_initial_state)
            G.load_state_dict(G_current_state)
            D.load_state_dict(D_current_syate)
            e_optimizer.load_state_dict(e_optimizer_initial_state)

    print()
    save_checkpoint(G, g_optimizer, args.wgan_epochs, avg_gloss, os.path.join(checkpoint_dir, 'G_checkpoint_final.pth'))
    save_checkpoint(D, d_optimizer, args.wgan_epochs, avg_dloss, os.path.join(checkpoint_dir, 'D_checkpoint_final.pth'))

    return losses

def train_d_step(G, D, g_optimizer, d_optimizer, x, adj, z, gumbell_type, device, lambda_gp):

    # real graphs loss
    real_logits, _ = D(x, adj)
    dloss_real = - torch.mean(real_logits)
    
    assert not torch.isnan(real_logits).any(), "NaN in real_logits"

    # generate graphs
    adj_logits, x_hat = G(z)
    adj_hat = process_adj(adj_logits, gumbell_type)

    assert not torch.isnan(adj_logits).any(), "NaN in adj_logits"
    assert not torch.isnan(x_hat).any(), "NaN in x_hat"
    
    # fake graphs loss
    fake_logits, _ = D(x_hat, adj_hat)
    dloss_fake = torch.mean(fake_logits)

    assert not torch.isnan(fake_logits).any(), "NaN in fake_logits"
    
    # gradient penalty from WGAN-GP
    # small adaptation -> macro penalty computed as sum of generated nodes and edges micro penalties
    eps = torch.rand(adj.shape[0], 1, 1).to(device)
    x_int0 = (eps * x + (1. - eps) * x_hat).requires_grad_(True)    # nodes
    x_int1 = (eps * adj + (1. - eps) * adj_hat).requires_grad_(True)    # edges
    grad0, grad1 = D(x_int0, x_int1)
    assert not torch.isnan(grad0).any(), "NaN in grad0"
    assert not torch.isnan(grad1).any(), "NaN in grad1"
    dloss_gp = gradient_penalty(grad0, x_int0, device) + gradient_penalty(grad1, x_int1, device)

    dloss = dloss_fake + dloss_real + lambda_gp * dloss_gp

    # backward and optimize
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    dloss.backward()
    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10.0)
    d_optimizer.step()

    return dloss

def train_g_step(G, D, g_optimizer, d_optimizer, z, gumbell_type, epoch, dloss):

    adj_logits, x_hat = G(z)
    adj_hat = process_adj(adj_logits, gumbell_type)

    fake_logits, _ = D(x_hat, adj_hat)
    gloss = - torch.mean(fake_logits)

    log_graph(adj_hat.cpu().detach().numpy(), gloss.item(), dloss.item(), epoch)

    # backward and optimize
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    gloss.backward()

    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=10.0)
    
    g_optimizer.step()

    return gloss