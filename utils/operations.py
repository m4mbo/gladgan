import numpy as np
import torch
import os
import pickle
import torch.nn as nn

def sample_z(batch_size, z_dim):
    return torch.tensor(np.random.normal(0, 1, size=(batch_size, z_dim)), dtype=torch.float32)

def update_lr(g_optimizer, d_optimizer, g_lr, d_lr):
    """
    Decay learning rates of the generator and discriminator.
    """
    for param_group in g_optimizer.param_groups:
        param_group['lr'] = g_lr
    for param_group in d_optimizer.param_groups:
        param_group['lr'] = d_lr

def gradient_penalty(y, x, device):
    """
    Compute gradient penalty: (L2_norm(dy/dx) - 1)**2.
    """
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save the model and optimizer state, along with the current epoch and loss.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    """
    Load the model and optimizer state from a checkpoint file.
    """
    checkpoint = torch.load(filepath, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def load_scores_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            auc_scores = pickle.load(file)
    else:
        auc_scores = {}
    return auc_scores

def save_scores_to_file(file_path, epoch, auc_score):
    auc_scores = load_scores_from_file(file_path)
    auc_scores[epoch] = auc_score
    with open(file_path, 'wb') as file:
        pickle.dump(auc_scores, file)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def track_gradients(module):
    def hook(grad):
        print(f'Gradient for {module}:\n {grad}')
    return hook